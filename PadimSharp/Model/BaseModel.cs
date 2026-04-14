using Padim.Data;
using Padim.Modules;
using Padim.Utils;
using System.Text;
using TorchSharp;
using TorchSharp.Modules;

namespace Padim.Model
{
    public class BaseModel
    {
        private readonly Config.Config config;
        private torch.nn.Module<torch.Tensor, (torch.Tensor, torch.Tensor, torch.Tensor)> net;
        private int totalDimension, subspaceDimension;

        private torch.Tensor idx, mean, cov;
        private float image_threshold, pixel_threshold;

        public BaseModel(Config.Config config)
        {
            this.config = config;

            switch (config.BackboneType)
            {
                case BackboneType.ResNet18:
                    {
                        net = Modules.ResNet.ResNet18();
                        this.totalDimension = 448;
                        this.subspaceDimension = 100;
                        break;
                    }
                case BackboneType.WideResNet50_2:
                    {
                        net = Modules.ResNet.WideResNet50_2();
                        this.totalDimension = 1792;
                        this.subspaceDimension = 550;
                        break;
                    }

                default:
                    {
                        throw new NotImplementedException($"Backbone type {config.BackboneType} is not implemented.");
                    }
            }

            net.load(config.PretrainedModelPath).to(config.Device);
            net.eval();
        }

        public void Train()
        {
            Console.WriteLine(config.ToString());
            WriteConfig();

            MVTecDataset.TrainDataset mvtecDataset = new MVTecDataset.TrainDataset(config.RootPath, config.ResizedWidth, config.ResizedHight, config.CroppedWidth, config.CroppedHeight);
            DataLoader trainLoader = new DataLoader(mvtecDataset, config.BatchSize, device: config.Device, num_worker: config.Workers);

            MVTecDataset.ValDataset valDataset = new MVTecDataset.ValDataset(config.RootPath, config.ResizedWidth, config.ResizedHight, config.CroppedWidth, config.CroppedHeight);
            DataLoader testLoader = new DataLoader(valDataset, config.BatchSize, device: config.Device, num_worker: config.Workers);

            Console.WriteLine("Start Training:");
            ulong seed = (ulong)DateTime.Now.Ticks;
            (float image_average_precision, float pixel_average_precision, float image_threshold, float pixel_threshold, torch.Tensor idx, torch.Tensor cov, torch.Tensor mean) = TrainEpoch(seed, trainLoader, testLoader);

            Console.WriteLine($"Image Average Precision: {image_average_precision}");
            Console.WriteLine($"Pixel Average Precision: {pixel_average_precision}");

            Weight weight = new Weight(this.subspaceDimension, height: config.CroppedHeight, width: config.CroppedWidth);
            string weightFolder = Path.Combine(config.OutputPath, "Weight");
            if (!Directory.Exists(weightFolder))
            {
                Directory.CreateDirectory(weightFolder);
            }
            weight.Save(Path.Combine(weightFolder, "weight.bin"), image_threshold, pixel_threshold, idx, cov, mean);
        }

        public (float image_average_precision, float pixel_average_precision, float image_threshold, float pixel_threshold, torch.Tensor idx, torch.Tensor cov, torch.Tensor mean) TrainEpoch(ulong seed, DataLoader trainLoader, DataLoader valLoader)
        {
            using (torch.no_grad())
            using (torch.NewDisposeScope())
            {
                List<torch.Tensor>[] opts = new List<torch.Tensor>[3];
                opts[0] = new List<torch.Tensor>();
                opts[1] = new List<torch.Tensor>();
                opts[2] = new List<torch.Tensor>();

                foreach (var trainBatch in trainLoader)
                {
                    using (torch.NewDisposeScope())
                    {
                        torch.Tensor inputs = trainBatch["image"].to(config.Dtype);
                        (torch.Tensor layer1, torch.Tensor layer2, torch.Tensor layer3) = net.forward(inputs);
                        opts[0].Add(layer1.MoveToOuterDisposeScope());
                        opts[1].Add(layer2.MoveToOuterDisposeScope());
                        opts[2].Add(layer3.MoveToOuterDisposeScope());
                    }
                }

                torch.Generator generator = new torch.Generator(seed);
                torch.Tensor idx = torch.randperm(this.totalDimension, generator: generator, dtype: torch.ScalarType.Int64);
                idx = idx.slice(0, 0, this.subspaceDimension, 1);
                idx = idx.to(config.Device);

                torch.Tensor embedding_vectors = Utils.Utils.GetEmbedding(opts);
                embedding_vectors = torch.index_select(embedding_vectors, 1, idx);
                long[] shape = embedding_vectors.shape;
                long B = shape[0], C = shape[1], H = shape[2], W = shape[3];
                embedding_vectors = embedding_vectors.view(B, C, H * W);

                torch.Tensor mean = torch.mean(embedding_vectors, dimensions: [0]);
                torch.Tensor cov = torch.zeros([C, C, H * W], device: config.Device);

                var I = torch.eye(C, device: config.Device);

                for (int i = 0; i < H * W; i++)
                {
                    var slice = embedding_vectors[torch.TensorIndex.Ellipsis, i];
                    var cov_slice = torch.cov(slice.T, correction: 0);
                    cov[torch.TensorIndex.Ellipsis, i] = cov_slice + 0.01 * I;
                }

                cov = cov.permute(2, 0, 1).inverse();

                List<torch.Tensor> tagTensors = new List<torch.Tensor>();
                List<torch.Tensor> truthTensors = new List<torch.Tensor>();

                List<torch.Tensor>[] valOpts = new List<torch.Tensor>[3];
                valOpts[0] = new List<torch.Tensor>();
                valOpts[1] = new List<torch.Tensor>();
                valOpts[2] = new List<torch.Tensor>();

                foreach (var testBatch in valLoader)
                {
                    using (torch.NewDisposeScope())
                    {
                        torch.Tensor inputs = testBatch["image"].to(config.Dtype);
                        tagTensors.Add(testBatch["tag"].clone().MoveToOuterDisposeScope());
                        truthTensors.Add(testBatch["truth"].clone().MoveToOuterDisposeScope());

                        (torch.Tensor layer1, torch.Tensor layer2, torch.Tensor layer3) = net.forward(inputs);
                        valOpts[0].Add(layer1.MoveToOuterDisposeScope());
                        valOpts[1].Add(layer2.MoveToOuterDisposeScope());
                        valOpts[2].Add(layer3.MoveToOuterDisposeScope());
                    }
                }


                torch.Tensor anomaly_map = Utils.Utils.ComputeAnomalyMap(valOpts, mean, cov, idx);
                torch.Tensor img_scores = torch.amax(anomaly_map, dims: [-2, -1]);
                var truthTensor = torch.concat(truthTensors);
                var tagTensor = torch.concat(tagTensors);

                //Console.WriteLine("Get image threshold and pixel threshold......");
                PrecisionRecall image_curve = new PrecisionRecall(tagTensor, img_scores);
                float image_threshold = image_curve.GetThreshold();
                (float image_precision, float image_recall, float f1) = image_curve.GetBestMetrics();
                float image_average_precision = image_curve.GetAveragePrecision();

                long size = anomaly_map.shape[0] * anomaly_map.shape[1] * anomaly_map.shape[2] * anomaly_map.shape[3];

                PrecisionRecall pixel_curve = new PrecisionRecall(truthTensor.view([size]), anomaly_map.view([size]));
                float pixel_threshold = pixel_curve.GetThreshold();
                //(float pixel_precision, float pixel_recall, float pixel_f1) = pixel_curve.GetBestMetrics();
                float pixel_average_precision = pixel_curve.GetAveragePrecision();

                this.idx = idx;
                this.mean = mean;
                this.cov = cov;
                this.image_threshold = image_threshold;
                this.pixel_threshold = pixel_threshold;

                return (image_average_precision, pixel_average_precision, image_threshold, pixel_threshold, idx.MoveToOuterDisposeScope(), cov.MoveToOuterDisposeScope(), mean.MoveToOuterDisposeScope());

            }
        }

        public void LoadWeight(string weightPath)
        {
            Weight weight = new Weight(this.subspaceDimension, config.CroppedWidth, config.CroppedHeight);
            weight.load(weightPath);

            this.idx = weight.get_buffer("idx")!.to(config.Device);
            this.mean = weight.get_buffer("mean")!.to(config.Device);
            this.cov = weight.get_buffer("cov")!.to(config.Device);
            this.image_threshold = weight.get_buffer("image_threshold")!.ToSingle();
            this.pixel_threshold = weight.get_buffer("pixel_threshold")!.ToSingle();
        }


        public (bool isGood, torch.Tensor image) Predict(string imagePath)
        {
            using (torch.no_grad())
            using (torch.NewDisposeScope())
            {
                torch.Tensor orgImg = torchvision.io.read_image(imagePath);
                torch.Tensor image = orgImg;
                torch.Tensor inputTensor = MVTecDataset.ProcessImage(image, config.ResizedWidth, config.ResizedHight, config.CroppedWidth, config.CroppedHeight);
                (torch.Tensor layer1, torch.Tensor layer2, torch.Tensor layer3) = net.forward(inputTensor.to(config.Device));
                List<torch.Tensor>[] opts = new List<torch.Tensor>[3];
                opts[0] = new List<torch.Tensor>() { layer1 };
                opts[1] = new List<torch.Tensor>() { layer2 };
                opts[2] = new List<torch.Tensor>() { layer3 };
                torch.Tensor anomaly_map = Utils.Utils.ComputeAnomalyMap(opts, this.mean, this.cov, this.idx).to(DeviceType.CPU);
                float max = anomaly_map.max().ToSingle();
                bool noGood = max > this.image_threshold;
                torch.Tensor imageTensor = orgImg.clone();

                if (noGood)
                {
                    int orgWidth = (int)orgImg.shape[2];
                    int orgHeight = (int)orgImg.shape[1];
                    int boundary = 6;
                    anomaly_map = (anomaly_map * (anomaly_map > this.pixel_threshold)).squeeze(0);
                    anomaly_map = torchvision.transforms.functional.resize(anomaly_map, orgHeight, orgWidth);

                    torch.Tensor boundaryTensor = torchvision.transforms.functional.resize(anomaly_map, orgHeight + 2 * boundary, orgWidth + 2 * boundary).crop(boundary, boundary, orgHeight, orgWidth);
                    torch.Tensor boundaryMask = (boundaryTensor > 0) ^ (anomaly_map > 0);
                    boundaryMask = torch.concat([boundaryMask, boundaryMask, boundaryMask]);

                    imageTensor[0] = (orgImg[0] + anomaly_map.squeeze(0) * 255.0f).clip(0, 255).to(torch.ScalarType.Byte);
                    imageTensor = (imageTensor + boundaryMask * 255.0f).clip(0, 255).to(torch.ScalarType.Byte);
                }
                return (!noGood, imageTensor.MoveToOuterDisposeScope());
            }
        }







        private void WriteConfig()
        {
            if (!Directory.Exists(config.OutputPath))
            {
                Directory.CreateDirectory(config.OutputPath);
            }
            string fileName = Path.Combine(config.OutputPath, "config.txt");
            StringBuilder stringBuilder = new StringBuilder();
            stringBuilder.AppendLine("Training Settings:");
            stringBuilder.AppendLine($"Date Time: {DateTime.Now}");
            stringBuilder.AppendLine(config.ToString());
            File.WriteAllText(fileName, stringBuilder.ToString());
        }



    }
}
