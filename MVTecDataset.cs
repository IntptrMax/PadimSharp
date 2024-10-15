using TorchSharp;
using static TorchSharp.torch;

namespace Padim
{
	internal class MVTecDataset
	{
		private static double[] means = [0.485, 0.456, 0.406], stdevs = [0.229, 0.224, 0.225];

		public class TrainDataset : torch.utils.data.Dataset
		{
			private long count = 0;
			private string[] files = new string[0];
			private int resizeWidth = 256;
			private int resizeHeight = 256;
			private int cropWidth = 224;
			private int cropHeight = 224;

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
			}

			public override long Count => this.count;
			public override Dictionary<string, Tensor> GetTensor(long index)
			{
				string file = files[(int)index];
				torchvision.io.DefaultImager = new torchvision.io.SkiaImager();

				int tag = (Directory.GetParent(file).Name.ToLower() == "good") ? 0 : 1;

				var transformers = torchvision.transforms.Compose([
					torchvision.transforms.Resize(resizeHeight,resizeWidth),
				torchvision.transforms.CenterCrop(cropHeight,cropWidth),
				torchvision.transforms.Normalize(means, stdevs)]);

				Tensor imgTensor = torchvision.io.read_image(file) / 255.0f;

				imgTensor = transformers.call(imgTensor.unsqueeze(0));
				var tensorDataDic = new Dictionary<string, Tensor>();
				tensorDataDic.Add("image", imgTensor.squeeze(0));
				tensorDataDic.Add("tag", torch.tensor(tag));

				return tensorDataDic;
			}

		}

		public class ValDataset : torch.utils.data.Dataset
		{
			private long count = 0;
			private string[] files = new string[0];
			private int resizeWidth = 256;
			private int resizeHeight = 256;
			private int cropWidth = 224;
			private int cropHeight = 224;

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
			}

			public override long Count => this.count;
			public override Dictionary<string, Tensor> GetTensor(long index)
			{
				string file = files[(int)index];
				torchvision.io.DefaultImager = new torchvision.io.SkiaImager();

				var parent = Directory.GetParent(file);
				int tag = (parent.Name.ToLower() == "good") ? 0 : 1;

				var truth_transformers = torchvision.transforms.Compose([
					torchvision.transforms.Resize(resizeHeight,resizeWidth),
				torchvision.transforms.CenterCrop(cropHeight,cropWidth)]);

				Tensor truthTensor = torch.zeros([1, cropHeight, cropWidth]);
				if (tag == 1)
				{
					string ground_truth_Path = Path.Combine(parent.Parent.Parent.FullName, "ground_truth", parent.Name, Path.GetFileNameWithoutExtension(file) + "_mask" + Path.GetExtension(file));
					truthTensor = torchvision.io.read_image(ground_truth_Path);
					truthTensor = truth_transformers.call(truthTensor) / 255.0f;
				}

				var transformers = torchvision.transforms.Compose([
					torchvision.transforms.Resize(resizeHeight,resizeWidth),
				torchvision.transforms.CenterCrop(cropHeight,cropWidth),
				torchvision.transforms.Normalize(means, stdevs)]);

				Tensor img = torchvision.io.read_image(file);
				var imgTensor = img / 255.0f;
				imgTensor = transformers.call(imgTensor.unsqueeze(0));
				var tensorDataDic = new Dictionary<string, Tensor>();
				tensorDataDic.Add("image", imgTensor.squeeze(0));
				tensorDataDic.Add("tag", torch.tensor(tag));
				tensorDataDic.Add("truth", truthTensor);
				tensorDataDic.Add("orgImage", img);

				return tensorDataDic;
			}
		}


		private Tensor Letterbox(Tensor image, int targetWidth, int targetHeight)
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
			Tensor scaledImage = torchvision.transforms.functional.resize(image, scaledHeight, scaledWidth);

			// 创建一个全零的张量，用于填充
			Tensor paddedImage = zeros(new long[] { 3, targetHeight, targetWidth }, image.dtype, image.device);

			// 将缩放后的图像放置在填充后的图像中心
			paddedImage[TensorIndex.Ellipsis, padTop..(padTop + scaledHeight), padLeft..(padLeft + scaledWidth)].copy_(scaledImage);

			GC.Collect();

			return paddedImage;
		}
	}
}
