using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Padim
{
	internal class ModelHelper
	{
		int width = 224;
		int height = 224;
		Module<Tensor, Tensor> model;
		private class TempTensor
		{
			public string Name;
			public float[] Data;
			public long[] Shape;
		}
		public ModelHelper(Module<Tensor, Tensor> model, int width = 224, int height = 224)
		{
			this.model = model;
			this.width = width;
			this.height = height;
		}

		public List<(string, Tensor)> Forward(Tensor input)
		{
			List<(string, Tensor)> outputs = new List<(string, Tensor)>();
			List<TempTensor> tempTensors = new List<TempTensor>();
			foreach (var named_module in model.named_children())
			{
				string name = named_module.name;
				if (name == "layer1" || name == "layer2" || name == "layer3")
				{
					((Sequential)named_module.module).register_forward_hook((Module, input, output) =>
					{
						tempTensors.Add(new TempTensor
						{
							Data = output.data<float>().ToArray(),
							Name = name,
							Shape = output.shape,
						});
						return null;
					});
				}
			}
			model.forward(input);

			var layer1output = tempTensors.Find(a => a.Name == "layer1");
			var layer2output = tempTensors.Find(a => a.Name == "layer2");
			var layer3output = tempTensors.Find(a => a.Name == "layer3");

			Tensor l1 = torch.tensor(layer1output.Data, layer1output.Shape, device: input.device);
			Tensor l2 = torch.tensor(layer2output.Data, layer2output.Shape, device: input.device);
			Tensor l3 = torch.tensor(layer3output.Data, layer3output.Shape, device: input.device);
			outputs.Add(new("layer1", l1));
			outputs.Add(new("layer2", l2));
			outputs.Add(new("layer3", l3));
			GC.Collect();
			return outputs;
		}

		public Tensor GetEmbedding(List<(string, Tensor)> outputs)
		{
			var t1s = outputs.FindAll(a => a.Item1 == "layer1");
			var t2s = outputs.FindAll(a => a.Item1 == "layer2");
			var t3s = outputs.FindAll(a => a.Item1 == "layer3");

			List<Tensor> layer1out = new List<Tensor>();
			List<Tensor> layer2out = new List<Tensor>();
			List<Tensor> layer3out = new List<Tensor>();
			for (int i = 0; i < t1s.Count; i++)
			{
				layer1out.Add(t1s[i].Item2);
				layer2out.Add(t2s[i].Item2);
				layer3out.Add(t3s[i].Item2);
			}

			var t1 = torch.concat(layer1out);
			var t2 = torch.concat(layer2out);
			var t3 = torch.concat(layer3out);

			Tensor embedding_vectors = EmbeddingConcat([t1, t2, t3]);
			return embedding_vectors;
		}

		private Tensor EmbeddingConcat(Tensor[] features)
		{
			var embeddings = features[0];

			for (int i = 1; i < features.Length; i++)
			{
				var layerEmbedding = features[i];
				layerEmbedding = torch.nn.functional.interpolate(layerEmbedding, size: [embeddings.shape[2], embeddings.shape[2]], mode: InterpolationMode.Nearest);
				embeddings = torch.cat([embeddings, layerEmbedding], 1);
			}
			return embeddings;
		}

		private Tensor ComputeDistance(Tensor embedding, Tensor mean, Tensor covariance)
		{
			long batch = embedding.shape[0];
			long channel = embedding.shape[1];
			long height = embedding.shape[2];
			long width = embedding.shape[3];

			Tensor inv_covariance = covariance.permute(2, 0, 1).inverse();
			var embedding_reshaped = embedding.reshape(batch, channel, height * width);
			var delta = (embedding_reshaped - mean).permute(2, 0, 1);
			var distances = (torch.matmul(delta, inv_covariance) * delta).sum(2).permute(1, 0);
			distances = distances.reshape(batch, 1, height, width);
			distances = distances.clamp(0).sqrt();
			return distances;
		}

		private Tensor UpSample(Tensor distance)
		{
			return torch.nn.functional.interpolate(distance, size: [height, width], mode: InterpolationMode.Bilinear, align_corners: false);
		}

		private Tensor SmoothAnomalyMap(Tensor anomalyMap, int sigma = 4)
		{
			int kernelSize = 2 * (int)(4.0 * sigma + 0.5) + 1;
			return torchvision.transforms.functional.gaussian_blur(anomalyMap, kernelSize, sigma);
		}

		public Tensor ComputeAnomalyMapInternal(Tensor embedding, Tensor mean, Tensor covariance)
		{
			var scoreMap = ComputeDistance(embedding, mean, covariance);
			var upSampledScoreMap = UpSample(scoreMap);
			var smoothedAnomalyMap = SmoothAnomalyMap(upSampledScoreMap);
			return smoothedAnomalyMap;
		}

		public Tensor ComputeAnomalyMap(List<(string, Tensor)> outputs, Tensor mean, Tensor covariance, Tensor idx)
		{
			Tensor embedding = GetEmbedding(outputs);
			var embeddingVectors = torch.index_select(embedding, 1, idx);
			return ComputeAnomalyMapInternal(embeddingVectors, mean, covariance);
		}

	}
}
