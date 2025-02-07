using TorchSharp;
using static TorchSharp.torch;

namespace Padim
{
	internal class ModelHelper
	{
		public static Tensor GetEmbedding(List<(string, Tensor)> outputs)
		{
			using (NewDisposeScope())
			{
				List<Tensor> layer1out = outputs.Where(a => a.Item1 == "layer1").Select(a => a.Item2).ToList();
				List<Tensor> layer2out = outputs.Where(a => a.Item1 == "layer2").Select(a => a.Item2).ToList();
				List<Tensor> layer3out = outputs.Where(a => a.Item1 == "layer3").Select(a => a.Item2).ToList();

				using Tensor t1 = torch.concat(layer1out);
				using Tensor t2 = torch.concat(layer2out);
				using Tensor t3 = torch.concat(layer3out);

				return EmbeddingConcat([t1, t2, t3]).MoveToOuterDisposeScope();
			}
		}

		private static Tensor EmbeddingConcat(Tensor[] features)
		{
			using (NewDisposeScope())
			{
				Tensor embeddings = features[0];
				embeddings = features.Skip(1)
									 .Select(layerEmbedding => nn.functional.interpolate(layerEmbedding, size: new long[] { embeddings.shape[2], embeddings.shape[2] }, mode: InterpolationMode.Nearest))
									 .Aggregate(embeddings, (current, layerEmbedding) => torch.cat(new[] { current, layerEmbedding }, 1));
				return embeddings.MoveToOuterDisposeScope();
			}
		}

		private static Tensor ComputeDistance(Tensor embedding, Tensor mean, Tensor covariance)
		{
			using (NewDisposeScope())
			{
				long batch = embedding.shape[0];
				long channel = embedding.shape[1];
				long height = embedding.shape[2];
				long width = embedding.shape[3];

				var embedding_reshaped = embedding.reshape(batch, channel, height * width);
				var delta = (embedding_reshaped - mean).permute(2, 0, 1);
				var distances = (torch.matmul(delta, covariance) * delta).sum(2).permute(1, 0);
				distances = distances.reshape(batch, 1, height, width);
				distances = distances.clamp(0).sqrt();
				return distances.MoveToOuterDisposeScope();
			}
		}

		private static Tensor UpSample(Tensor distance, int height = 224, int width = 224)
		{
			return torch.nn.functional.interpolate(distance, size: [height, width], mode: InterpolationMode.Bilinear, align_corners: false);
		}

		private static Tensor SmoothAnomalyMap(Tensor anomalyMap, int sigma = 4)
		{
			int kernelSize = 2 * (int)(4.0 * sigma + 0.5) + 1;
			return torchvision.transforms.functional.gaussian_blur(anomalyMap, kernelSize, sigma);
		}

		public static Tensor ComputeAnomalyMapInternal(Tensor embedding, Tensor mean, Tensor covariance)
		{
			using (NewDisposeScope())
			{
				var scoreMap = ComputeDistance(embedding, mean, covariance);
				var upSampledScoreMap = UpSample(scoreMap);
				var smoothedAnomalyMap = SmoothAnomalyMap(upSampledScoreMap);
				return smoothedAnomalyMap.MoveToOuterDisposeScope();
			}
		}

		public static Tensor ComputeAnomalyMap(List<(string, Tensor)> outputs, Tensor mean, Tensor covariance, Tensor idx)
		{
			using (NewDisposeScope())
			{
				Tensor embedding = GetEmbedding(outputs);
				Tensor embeddingVectors = torch.index_select(embedding, 1, idx);
				return ComputeAnomalyMapInternal(embeddingVectors, mean, covariance).MoveToOuterDisposeScope();
			}
		}
	}
}
