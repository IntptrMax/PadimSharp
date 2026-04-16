using TorchSharp;

namespace PadimSharp.Data
{
    internal class MVTecDataset
    {
        private static double[] means = [0.485, 0.456, 0.406], stdevs = [0.229, 0.224, 0.225];

        public class TrainDataset : torch.utils.data.Dataset
        {
            private long count = 0;
            private string[] files = new string[0];
            private int resizedWidth = 256;
            private int resizedHeight = 256;
            private int croppedWidth = 224;
            private int croppedHeight = 224;

            public TrainDataset(string rootPath, int resizeWidth = 256, int resizeHeight = 256, int cropWidth = 224, int cropHeight = 224)
            {
                string path = Path.Combine(rootPath, "train");
                string[] files = Directory.GetFiles(path, "*.*", SearchOption.AllDirectories).Where(file =>
                {
                    string extension = Path.GetExtension(file).ToLower();
                    return (extension == ".jpg" || extension == ".png" || extension == ".bmp");
                }).ToArray();
                this.files = files;
                this.count = files.Length;
                this.resizedWidth = resizeWidth;
                this.resizedHeight = resizeHeight;
                this.croppedHeight = cropHeight;
                this.croppedWidth = cropWidth;
            }

            public override long Count => this.count;
            public override Dictionary<string, torch.Tensor> GetTensor(long index)
            {
                string file = files[(int)index];
                torchvision.io.DefaultImager = new torchvision.io.SkiaImager();

                int tag = (Directory.GetParent(file)!.Name.ToLower() == "good") ? 0 : 1;

                var transformers = torchvision.transforms.Compose([
                    torchvision.transforms.Resize(resizedHeight, resizedWidth),
                    torchvision.transforms.CenterCrop(croppedHeight, croppedWidth),
                    torchvision.transforms.Normalize(means, stdevs)]);

                torch.Tensor imgTensor = torchvision.io.read_image(file) / 255.0f;

                imgTensor = transformers.call(imgTensor.unsqueeze(0));
                var tensorDataDic = new Dictionary<string, torch.Tensor>();
                tensorDataDic.Add("image", imgTensor.squeeze(0));
                tensorDataDic.Add("tag", torch.tensor(tag));

                return tensorDataDic;
            }

        }

        public class ValDataset : torch.utils.data.Dataset
        {
            private long count = 0;
            private string[] files = new string[0];
            private int resizedWidth = 256;
            private int resizedHeight = 256;
            private int croppedWidth = 224;
            private int croppedHeight = 224;

            public ValDataset(string rootPath, int resizeWidth = 256, int resizeHeight = 256, int cropWidth = 224, int cropHeight = 224)
            {
                string path = Path.Combine(rootPath, "test");
                string[] files = Directory.GetFiles(path, "*.*", SearchOption.AllDirectories).Where(file =>
                {
                    string extension = Path.GetExtension(file).ToLower();
                    return (extension == ".jpg" || extension == ".png" || extension == ".bmp");
                }).ToArray();
                this.files = files;
                this.count = files.Length;
                this.resizedWidth = resizeWidth;
                this.resizedHeight = resizeHeight;
                this.croppedHeight = cropHeight;
                this.croppedWidth = cropWidth;
            }

            public override long Count => this.count;
            public override Dictionary<string, torch.Tensor> GetTensor(long index)
            {
                string file = files[(int)index];
                torchvision.io.DefaultImager = new torchvision.io.SkiaImager();

                var parent = Directory.GetParent(file);
                int tag = (parent!.Name.ToLower() == "good") ? 0 : 1;

                var truth_transformers = torchvision.transforms.Compose([
                    torchvision.transforms.Resize(resizedHeight,resizedWidth),
                    torchvision.transforms.CenterCrop(croppedHeight,croppedWidth),
			        torchvision.transforms.GaussianBlur(5,5)
                ]);

                torch.Tensor truthTensor = torch.zeros([1, croppedHeight, croppedWidth]);
                if (tag == 1)
                {
                    string ground_truth_Path = Path.Combine(parent.Parent!.Parent!.FullName, "ground_truth", parent.Name, Path.GetFileNameWithoutExtension(file) + "_mask" + Path.GetExtension(file));
                    if (File.Exists(ground_truth_Path))
                    {
                        truthTensor = torchvision.io.read_image(ground_truth_Path);
                        truthTensor = truth_transformers.call(truthTensor) / 255.0f;
                    }
                    else
                    {
                        truthTensor = torch.ones([1, croppedHeight, croppedWidth], torch.ScalarType.Float32);
                    }
                }

                var transformers = torchvision.transforms.Compose([
                    torchvision.transforms.Resize(resizedHeight,resizedWidth),
                    torchvision.transforms.CenterCrop(croppedHeight,croppedWidth),
                    torchvision.transforms.Normalize(means, stdevs)]);

                torch.Tensor img = torchvision.io.read_image(file);
                var imgTensor = img / 255.0f;
                imgTensor = transformers.call(imgTensor.unsqueeze(0));
                var tensorDataDic = new Dictionary<string, torch.Tensor>();
                tensorDataDic.Add("image", imgTensor.squeeze(0));
                tensorDataDic.Add("tag", torch.tensor(tag));
                tensorDataDic.Add("truth", truthTensor);
                tensorDataDic.Add("orgImage", img);

                return tensorDataDic;
            }
        }


        private torch.Tensor Letterbox(torch.Tensor image, int targetWidth, int targetHeight)
        {
            // 获取图像的原始尺寸
            int originalWidth = (int)image.shape[2];
            int originalHeight = (int)image.shape[1];

            // 计算缩放比例
            float scale = Math.Min((float)targetWidth / originalWidth, (float)targetHeight / originalHeight);

            // 计算缩放后的尺寸
            int scaledWidth = (int)(originalWidth * scale);
            int scaledHeight = (int)(originalHeight * scale);

            // 计算填充后的尺寸
            int padLeft = (targetWidth - scaledWidth) / 2;
            int padRight = targetWidth - scaledWidth - padLeft;
            int padTop = (targetHeight - scaledHeight) / 2;
            int padBottom = targetHeight - scaledHeight - padTop;

            // 缩放图像
            torch.Tensor scaledImage = torchvision.transforms.functional.resize(image, scaledHeight, scaledWidth);

            // 创建一个全零的张量，用于填充
            torch.Tensor paddedImage = torch.zeros(new long[] { 3, targetHeight, targetWidth }, image.dtype, image.device);

            // 将缩放后的图像放置在填充后的图像中心
            paddedImage[torch.TensorIndex.Ellipsis, padTop..(padTop + scaledHeight), padLeft..(padLeft + scaledWidth)].copy_(scaledImage);

            GC.Collect();

            return paddedImage;
        }

        public static torch.Tensor ProcessImage(torch.Tensor image, int resizedWidth = 256, int resizedHeight = 256, int croppedWidth = 224, int croppedHeight = 224)
        {
            var transformers = torchvision.transforms.Compose([
                torchvision.transforms.Resize(resizedHeight,resizedWidth),
                    torchvision.transforms.CenterCrop(croppedHeight,croppedWidth),
                    torchvision.transforms.Normalize(means, stdevs)]);
            torch.Tensor imgTensor = image / 255.0f;
            imgTensor = transformers.call(imgTensor.unsqueeze(0));
            return imgTensor;
        }
    }
}
