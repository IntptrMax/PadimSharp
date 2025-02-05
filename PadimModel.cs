using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Padim
{
	internal class PadimModel : Module<Tensor, List<(string, Tensor)>>
	{
		private Conv2d conv1;
		private BatchNorm2d bn1;
		private ReLU relu;
		private MaxPool2d maxPool2D;

		private Sequential layer1;
		private Sequential layer2;
		private Sequential layer3;


		public PadimModel() : base("ResNet18")
		{
			// First conv layer
			conv1 = Conv2d(3, 64, kernelSize: 7, stride: 2, padding: 3, bias: false);
			bn1 = BatchNorm2d(64);
			relu = ReLU(inplace: true);
			maxPool2D = MaxPool2d(kernelSize: 3, stride: 2, padding: 1);

			// ResNet18's 3 layers
			layer1 = MakeLayer(64, 64, numBlocks: 2, stride: 1);
			layer2 = MakeLayer(64, 128, numBlocks: 2, stride: 2);
			layer3 = MakeLayer(128, 256, numBlocks: 2, stride: 2);
			RegisterComponents();

		}

		public override List<(string, Tensor)> forward(Tensor input)
		{
			using (NewDisposeScope())
			{
				Tensor x = conv1.forward(input);
				x = bn1.forward(x);
				x = relu.forward(x);
				x = maxPool2D.forward(x);
				var layer1_out = layer1.forward(x);
				var layer2_out = layer2.forward(layer1_out);
				var layer3_out = layer3.forward(layer2_out);

				List<(string, Tensor)> outputs = new List<(string, Tensor)>();
				outputs.Add(("layer1", layer1_out.MoveToOuterDisposeScope()));
				outputs.Add(("layer2", layer2_out.MoveToOuterDisposeScope()));
				outputs.Add(("layer3", layer3_out.MoveToOuterDisposeScope()));
				return outputs;
			}
		}


		private Sequential MakeLayer(int inChannels, int outChannels, int numBlocks, int stride)
		{
			var layers = Sequential();
			layers.append(new BasicBlock(inChannels, outChannels, stride));
			for (int i = 1; i < numBlocks; i++)
			{
				layers.append(new BasicBlock(outChannels, outChannels, stride: 1));
			}
			return layers;
		}

		private class BasicBlock : Module<Tensor, Tensor>
		{
			private readonly Conv2d conv1;
			private readonly BatchNorm2d bn1;
			private readonly Conv2d conv2;
			private readonly BatchNorm2d bn2;
			private readonly Sequential downsample;

			public BasicBlock(int inChannels, int outChannels, int stride) : base("BasicBlock")
			{
				conv1 = Conv2d(inChannels, outChannels, kernelSize: 3, stride: stride, padding: 1, bias: false);
				bn1 = BatchNorm2d(outChannels);

				conv2 = Conv2d(outChannels, outChannels, kernelSize: 3, stride: 1, padding: 1, bias: false);
				bn2 = BatchNorm2d(outChannels);

				if (inChannels != outChannels || stride != 1)
				{
					downsample = Sequential(
						Conv2d(inChannels, outChannels, kernelSize: 1, stride: stride, bias: false),
						BatchNorm2d(outChannels)
					);
				}
				else
				{
					downsample = Sequential();
				}

				RegisterComponents();
			}

			public override Tensor forward(Tensor x)
			{
				using (NewDisposeScope())
				{
					var identity = x;

					var output = conv1.forward(x);
					output = bn1.forward(output);
					output = ReLU().forward(output);

					output = conv2.forward(output);
					output = bn2.forward(output);

					if (downsample != null)
					{
						identity = downsample.forward(x);
					}

					output += identity;
					output = ReLU().forward(output);

					return output.MoveToOuterDisposeScope();
				}
			}
		}
	}
}
