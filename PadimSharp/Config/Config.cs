using Padim.Modules;
using System.Text;
using TorchSharp;

namespace Padim.Config
{
    public class Config
    {
        /// <summary>
        /// Data root path
        /// </summary>
        public string RootPath { get; set; } = @"..\..\..\Assets\MVTecDataset";

        /// <summary>
        /// Weights and logs output path
        /// </summary>
        public string OutputPath { get; set; } = @".\Output";

        /// <summary>
        /// Train resized image width
        /// </summary>
        public int ResizedWidth { get; set; } = 256;

        /// <summary>
        /// Train resized image height
        /// </summary>
        public int ResizedHight { get; set; } = 256;

        /// <summary>
        /// Train cropped image width
        /// </summary>
        public int CroppedWidth { get; set; } = 224;

        /// <summary>
        /// Train cropped image height
        /// </summary>
        public int CroppedHeight { get; set; } = 224;

        /// <summary>
        /// Train batch size
        /// </summary>
        public int BatchSize { get; set; } = 16;

        /// <summary>
        /// Train epochs
        /// </summary>
        public int Epochs { get; set; } = 100;

        /// <summary>
        /// Workers for training
        /// </summary>
        public int Workers { get; set; } = Math.Min(Environment.ProcessorCount / 2, 4);

        /// <summary>
        /// Device for Yolo running, can be CPU or Cuda
        /// </summary>
        public DeviceType DeviceType { get; set; } = DeviceType.CUDA;

        /// <summary>
        /// Scalar Type for Yolo running, can be Float32, Float16, BFloat16
        /// </summary>
        public torch.ScalarType ScalarType { get; set; } = torch.ScalarType.Float32;

        /// <summary>
        /// Backbone type for Unet, can be ResNet18. ResNet18 is the default backbone and also the recommended backbone for most cases. 
        /// </summary>
        public BackboneType BackboneType { get; set; } = BackboneType.ResNet18;

        /// <summary>
        /// Pretrained model path. If the backbone is ResNet18, the pretrained model should be ResNet18 pretrained on ImageNet. The default value is the path to the ResNet18 pretrained model provided in this project.
        /// </summary>
        public string PretrainedModelPath { get; set; } = @"..\..\..\Assets\Models\resnet18.bin";

        public Config(string? rootPath = null, string? outputPath = null, int? resizedWidth = null, int? resizedHeight = null,
            int? croppedWidth = null, int? croppedHeight = null, int? batchSize = null, BackboneType? backboneType = null,
            string? pretrainedModelPath = null, int? epochs = null, int? workers = null, DeviceType? deviceType = null,
            torch.ScalarType? dtype = null)
        {
            this.RootPath = rootPath ?? RootPath;
            this.OutputPath = string.IsNullOrEmpty(outputPath) ? @".\output" : outputPath;
            this.ResizedWidth = resizedWidth ?? ResizedWidth;
            this.ResizedHight = resizedHeight ?? ResizedHight;
            this.CroppedWidth = croppedWidth ?? CroppedWidth;
            this.CroppedHeight = croppedHeight ?? CroppedHeight;
            this.BatchSize = batchSize ?? BatchSize;
            this.Epochs = epochs ?? Epochs;
            this.Workers = workers ?? Math.Min(Environment.ProcessorCount / 2, 4);
            this.DeviceType = deviceType ?? DeviceType;
            this.ScalarType = dtype ?? ScalarType;
            this.BackboneType = backboneType ?? BackboneType;
            this.PretrainedModelPath = pretrainedModelPath ?? PretrainedModelPath;
        }

        public torch.Device Device => new torch.Device((TorchSharp.DeviceType)DeviceType);
        public torch.ScalarType Dtype => (torch.ScalarType)ScalarType;

        public override string ToString()
        {
            StringBuilder stringBuilder = new StringBuilder();
            stringBuilder.AppendLine($"Precision type: {Dtype}");
            stringBuilder.AppendLine($"Device type: {Device}");
            stringBuilder.AppendLine($"Image Width: {ResizedWidth}");
            stringBuilder.AppendLine($"Image Height: {ResizedHight}");
            stringBuilder.AppendLine($"Crop Width: {CroppedWidth}");
            stringBuilder.AppendLine($"Crop Height: {CroppedHeight}");
            stringBuilder.AppendLine($"Epochs: {Epochs}");
            stringBuilder.AppendLine($"Batch Size: {BatchSize}");
            stringBuilder.AppendLine($"Num Workers: {Workers}");
            stringBuilder.AppendLine($"Backbone Type: {BackboneType}");
            stringBuilder.AppendLine($"Pretrained Model Path: \"{Path.GetFullPath(PretrainedModelPath)}\"");
            stringBuilder.AppendLine($"Root Path: \"{Path.GetFullPath(RootPath)}\"");
            stringBuilder.AppendLine($"Output Path: \"{Path.GetFullPath(OutputPath)}\"");
            return stringBuilder.ToString();
        }
    }
}
