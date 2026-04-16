using PadimSharp.Config;
using PadimSharp.Model;
using System.Diagnostics;
using TorchSharp;

namespace PadimSharpDemo
{
    internal class Program
    {
        static void Main(string[] args)
        {
            DeviceType deviceType = DeviceType.CUDA;
            torch.ScalarType scalarType = torch.ScalarType.Float32;
            PadimSharp.Modules.BackboneType backboneType = PadimSharp.Modules.BackboneType.ResNet18;  // WideResNet50_2 is better but slower, and need more GPU memory.
            string dataPath = @"..\..\..\Assets\MVTecDataset";
            string modelPath = @"..\..\..\Assets\Models\ResNet18.bin";
            string outputPath = @".\Output";
            int resizeSize = 256;
            int cropSize = 224;
            int workers = 4;
            int batchSize = 16;

            Config config = new Config()
            {
                RootPath = dataPath,
                OutputPath = outputPath,
                ResizedHight = resizeSize,
                ResizedWidth = resizeSize,
                CroppedHeight = cropSize,
                CroppedWidth = cropSize,
                BatchSize = batchSize,
                Workers = workers,
                ScalarType = scalarType,
                PretrainedModelPath = modelPath,
                BackboneType = backboneType,
                DeviceType = deviceType
            };

            BaseModel model = new BaseModel(config);
            model.Train();

            Test(model, outputPath, dataPath);
        }

        private static void Test(BaseModel model, string outputPath, string dataPath)
        {
            Console.WriteLine();
            Console.WriteLine("Start Testing:");
            string weightPath = Path.Combine(outputPath, "weight", "weight.bin");
            model.LoadWeight(weightPath);
            Stopwatch stopwatch = Stopwatch.StartNew();

            string[] files = Directory.GetFiles(Path.Combine(dataPath, "test"), "*.png", SearchOption.AllDirectories);
            int uncheckedCount = 0, wrongCount = 0, goodCount = 0, checkedCount = 0;
            for (int i = 0; i < files.Length; i++)
            {
                string predictImagePath = files[i];
                string name = Directory.GetParent(predictImagePath)!.Name;
                bool good = name.ToLower().Trim().Equals("good");
                (bool predictGood, torch.Tensor image) = model.Predict(predictImagePath);
                if (predictGood)
                {
                    if (good)
                    {
                        if (!Directory.Exists("temp/good"))
                        {
                            Directory.CreateDirectory("temp/good");
                        }
                        torchvision.io.write_image_async(image, "temp/good/" + i.ToString("000") + "_result.jpg", torchvision.ImageFormat.Jpeg);
                        goodCount++;
                    }
                    else
                    {
                        if (!Directory.Exists("temp/unchecked"))
                        {
                            Directory.CreateDirectory("temp/unchecked");
                        }
                        torchvision.io.write_image_async(image, "temp/unchecked/" + i.ToString("000") + "_result.jpg", torchvision.ImageFormat.Jpeg);
                        uncheckedCount++;
                    }
                }
                else
                {
                    if (good)
                    {
                        if (!Directory.Exists("temp/wrong"))
                        {
                            Directory.CreateDirectory("temp/wrong");
                        }
                        torchvision.io.write_image_async(image, "temp/wrong/" + i.ToString("000") + "_result.jpg", torchvision.ImageFormat.Jpeg);
                        wrongCount++;
                    }
                    else
                    {
                        if (!Directory.Exists("temp/checked"))
                        {
                            Directory.CreateDirectory("temp/checked");
                        }
                        torchvision.io.write_image_async(image, "temp/checked/" + i.ToString("000") + "_result.jpg", torchvision.ImageFormat.Jpeg);
                        checkedCount++;
                    }
                }
            }

            Console.WriteLine($"Test count:{wrongCount + checkedCount + uncheckedCount + goodCount} wrong:{wrongCount} unchecked:{uncheckedCount}");
            Console.WriteLine($"Test time: {stopwatch.ElapsedMilliseconds}ms");
        }

    }
}
