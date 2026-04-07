using System.Diagnostics;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using Module = TorchSharp.torch.nn.Module;

namespace Padim
{
    internal class Program
    {
        private static Device device = new Device(DeviceType.CUDA);
        private static ScalarType scalarType = ScalarType.Float32;
        private static string dataPath = @"..\..\..\Assets\MVTecDataset";
        private static string modelPath = @"..\..\..\Assets\Models\resnet18.bin";
        private static string weightPath = "weight.bin";
        private static int t_d = 448;
        private static int d = 100;

        private static Modules.ResNet resnet;

        static void Main(string[] args)
        {
            // Load model
            Console.WriteLine("Load model......");
            resnet = Modules.ResNet.ResNet18();
            resnet.load(modelPath);

            resnet.to(device, scalarType);
            resnet.eval();

            Console.WriteLine();
            Stopwatch stopwatch = Stopwatch.StartNew();
            Train();
            Console.WriteLine($"Train time: {stopwatch.ElapsedMilliseconds}ms");
            Console.WriteLine();
            stopwatch.Restart();
            Val();
            Console.WriteLine($"Val time: {stopwatch.ElapsedMilliseconds}ms");
            stopwatch.Restart();
            Console.WriteLine();
            Test();
            Console.WriteLine($"Test time: {stopwatch.ElapsedMilliseconds}ms");
        }

        private static void Train()
        {
            Console.WriteLine("Train begin......");
            using (torch.no_grad())
            {
                Tensor idx = torch.randint(0, t_d, new long[] { d }, device: device, dtype: ScalarType.Int64);
                MVTecDataset.TrainDataset mvtecDataset = new MVTecDataset.TrainDataset(dataPath, 256, 256, 224, 224);
                DataLoader trainLoader = new DataLoader(mvtecDataset, 32, device: device, num_worker: 8);

                List<Tensor>[] opts = new List<Tensor>[3];
                opts[0] = new List<Tensor>();
                opts[1] = new List<Tensor>();
                opts[2] = new List<Tensor>();

                Console.WriteLine("Get layers......");

                foreach (var trainBatch in trainLoader)
                {
                    using (NewDisposeScope())
                    {
                        Tensor inputs = trainBatch["image"].to(scalarType, device);
                        (Tensor layer1, Tensor layer2, Tensor layer3) = resnet.forward(inputs);
                        opts[0].Add(layer1.MoveToOuterDisposeScope());
                        opts[1].Add(layer2.MoveToOuterDisposeScope());
                        opts[2].Add(layer3.MoveToOuterDisposeScope());
                    }
                }


                Tensor embedding_vectors = ModelHelper.GetEmbedding(opts);
                embedding_vectors = torch.index_select(embedding_vectors, 1, idx);
                long[] shape = embedding_vectors.shape;
                long B = shape[0], C = shape[1], H = shape[2], W = shape[3];
                embedding_vectors = embedding_vectors.view(B, C, H * W);

                Console.WriteLine("Get mean......");
                Tensor mean = torch.mean(embedding_vectors, dimensions: [0]);

                Console.WriteLine("Get cov......");
                Tensor cov = torch.zeros([C, C, H * W], device: device);

                var I = torch.eye(C, device: device);

                for (int i = 0; i < H * W; i++)
                {
                    var slice = embedding_vectors[TensorIndex.Ellipsis, i];
                    var cov_slice = torch.cov(slice.T, correction: 0);
                    cov[TensorIndex.Ellipsis, i] = cov_slice + 0.01 * I;
                }

                cov = cov.permute(2, 0, 1).inverse();

                Console.WriteLine("Save weight......");
                Weight weight = new Weight();
                weight.get_buffer("cov").copy_(cov);
                weight.get_buffer("mean").copy_(mean);
                weight.get_buffer("idx").copy_(idx);
                weight.save(weightPath);

                Console.WriteLine("Train done.");
                Console.WriteLine();
            }
        }

        private static void Val()
        {
            Console.WriteLine("Val begin......");
            Weight weight = new Weight();
            weight.load(weightPath);

            Tensor idx = weight.get_buffer("idx").to(device);
            Tensor mean = weight.get_buffer("mean").to(device);
            Tensor cov = weight.get_buffer("cov").to(device);

            MVTecDataset.ValDataset valDataset = new MVTecDataset.ValDataset(dataPath, 256, 256, 224, 224);
            DataLoader testLoader = new DataLoader(valDataset, 1, device: device, num_worker: 8);

            List<Tensor> tagTensors = new List<Tensor>();
            List<Tensor> truthTensors = new List<Tensor>();

            Console.WriteLine("Get layers......");

            List<Tensor>[] opts = new List<Tensor>[3];
            opts[0] = new List<Tensor>();
            opts[1] = new List<Tensor>();
            opts[2] = new List<Tensor>();

            using (torch.no_grad())
            {
                foreach (var testBatch in testLoader)
                {
                    Tensor inputs = testBatch["image"].to(scalarType, device);
                    tagTensors.Add(testBatch["tag"].clone());
                    truthTensors.Add(testBatch["truth"].clone());

                    (Tensor layer1, Tensor layer2, Tensor layer3) = resnet.forward(inputs);
                    opts[0].Add(layer1);
                    opts[1].Add(layer2);
                    opts[2].Add(layer3);
                }
            }

            Tensor anomaly_map = ModelHelper.ComputeAnomalyMap(opts, mean, cov, idx);
            Tensor img_scores = torch.amax(anomaly_map, dims: [-2, -1]);
            var truthTensor = torch.concat(truthTensors);
            var tagTensor = torch.concat(tagTensors);


            Console.WriteLine("Get image threshold and pixel threshold......");
            PrecisionRecall image_curve = new PrecisionRecall(tagTensor, img_scores);
            float image_threshold = image_curve.GetThreshold();

            long size = anomaly_map.shape[0] * anomaly_map.shape[1] * anomaly_map.shape[2] * anomaly_map.shape[3];

            PrecisionRecall pixel_curve = new PrecisionRecall(truthTensor.view([size]), anomaly_map.view([size]));
            float pixel_threshold = pixel_curve.GetThreshold();

            Console.WriteLine("Image threshold is: {0}", image_threshold);
            Console.WriteLine("Pixel threshold is: {0}", pixel_threshold);

            Console.WriteLine("Save weight......");
            weight.get_buffer("image_threshold").copy_(torch.tensor(image_threshold));
            weight.get_buffer("pixel_threshold").copy_(torch.tensor(pixel_threshold));
            weight.save(weightPath);
            Console.WriteLine("Val done.");
            Console.WriteLine();
        }

        private static void Test()
        {
            Console.WriteLine("Test begin......");

            Weight weight = new Weight();
            weight.load(weightPath);

            Tensor idx = weight.get_buffer("idx").to(device);
            Tensor mean = weight.get_buffer("mean").to(device);
            Tensor cov = weight.get_buffer("cov").to(device);
            float image_threshold = weight.get_buffer("image_threshold").ToSingle();
            float pixel_threshold = weight.get_buffer("pixel_threshold").ToSingle();

            string testDataPath = dataPath;
            MVTecDataset.ValDataset valDataset = new MVTecDataset.ValDataset(testDataPath, 256, 256, 224, 224);

            List<(string, Tensor)> outputs = new List<(string, Tensor)>();

            Console.WriteLine("Get layers......");

            int wrongCount = 0;
            int uncheckedCount = 0;
            using (torch.no_grad())
            {
                for (int i = 0; i < valDataset.Count; i++)
                {
                    var tensors = valDataset.GetTensor(i);
                    Tensor tagTensor = tensors["tag"].to(device);

                    (Tensor layer1, Tensor layer2, Tensor layer3) = resnet.forward(tensors["image"].to(scalarType, device).unsqueeze(0));

                    List<Tensor>[] opts = new List<Tensor>[] { new List<Tensor> { layer1 }, new List<Tensor> { layer2 }, new List<Tensor> { layer3 } };

                    Tensor anomaly_map = ModelHelper.ComputeAnomalyMap(opts, mean, cov, idx);

                    float max = anomaly_map.max().ToSingle();
                    bool noGood = max > image_threshold;
                    Console.WriteLine("Image threshold is {0} and score is {1}", image_threshold, max);
                    Console.WriteLine("Current image is good: {0}", !noGood);
                    Tensor orgImg = tensors["orgImage"].clone().to(device);

                    if (noGood)
                    {
                        Console.WriteLine("Get anomaly mask ......");

                        int orgWidth = (int)orgImg.shape[2];
                        int orgHeight = (int)orgImg.shape[1];
                        int boundary = 6;
                        anomaly_map = (anomaly_map * (anomaly_map > pixel_threshold)).squeeze(0);
                        anomaly_map = torchvision.transforms.functional.resize(anomaly_map, orgHeight, orgWidth);

                        Tensor boundaryTensor = torchvision.transforms.functional.resize(anomaly_map, orgHeight + 2 * boundary, orgWidth + 2 * boundary).crop(boundary, boundary, orgHeight, orgWidth);
                        Tensor boundaryMask = (boundaryTensor > 0) ^ (anomaly_map > 0);
                        boundaryMask = torch.concat([boundaryMask, boundaryMask, boundaryMask]);

                        var imageTensor = orgImg.clone();
                        imageTensor[0] = (orgImg[0] + anomaly_map.squeeze(0) * 255.0f).clip(0, 255).to(ScalarType.Byte);
                        imageTensor = (imageTensor + boundaryMask * 255.0f).clip(0, 255).to(ScalarType.Byte);

                        if (tagTensor.ToInt64() == 1)
                        {
                            if (!Directory.Exists("temp/checked"))
                            {
                                Directory.CreateDirectory("temp/checked");
                            }
                            torchvision.io.write_image_async(imageTensor.cpu(), "temp/checked/" + i + "_result.jpg", torchvision.ImageFormat.Jpeg);
                        }
                        else
                        {
                            if (!Directory.Exists("temp/wrong"))
                            {
                                Directory.CreateDirectory("temp/wrong");
                            }
                            torchvision.io.write_image_async(imageTensor.cpu(), "temp/wrong/" + i + "_result.jpg", torchvision.ImageFormat.Jpeg);
                            wrongCount++;
                        }
                    }
                    else
                    {
                        if (tagTensor.ToInt64() == 1)
                        {
                            if (!Directory.Exists("temp/unchecked"))
                            {
                                Directory.CreateDirectory("temp/unchecked");
                            }
                            torchvision.io.write_image_async(orgImg.cpu(), "temp/unchecked/" + i + "_result.jpg", torchvision.ImageFormat.Jpeg);
                            uncheckedCount++;
                        }
                        else
                        {
                            if (!Directory.Exists("temp/good"))
                            {
                                Directory.CreateDirectory("temp/good");
                            }
                            torchvision.io.write_image_async(orgImg.cpu(), "temp/good/" + i + "_result.jpg", torchvision.ImageFormat.Jpeg);
                        }

                    }

                    GC.Collect();
                }
            }

            Console.WriteLine();
            Console.WriteLine("Test count: " + valDataset.Count + " wrong: " + wrongCount + " unchecked: " + uncheckedCount);
            Console.WriteLine("Test done.");
        }



        public class Weight : Module
        {
            public Weight(int d = 100, int height = 224, int width = 224) : base("profile")
            {
                int sz = height * width / 16;
                this.register_buffer("cov", torch.zeros(sz, d, d));
                this.register_buffer("mean", torch.zeros(d, sz));
                this.register_buffer("image_threshold", torch.zeros(1));
                this.register_buffer("pixel_threshold", torch.zeros(1));
                this.register_buffer("idx", torch.zeros(d, ScalarType.Int32));
            }
        }


    }
}
