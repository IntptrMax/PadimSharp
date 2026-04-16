using TorchSharp;

namespace PadimSharp.Utils
{
    internal class Utils
    {
        public static torch.Tensor GetEmbedding(List<torch.Tensor>[] outputs)
        {
            using (torch.no_grad())
            using (torch.NewDisposeScope())
            {
                torch.Tensor t1 = torch.concat(outputs[0]);
                torch.Tensor t2 = torch.concat(outputs[1]);
                torch.Tensor t3 = torch.concat(outputs[2]);

                return EmbeddingConcat([t1, t2, t3]).MoveToOuterDisposeScope();
            }
        }

        private static torch.Tensor EmbeddingConcat(torch.Tensor[] features)
        {
            using (torch.no_grad())
            using (torch.NewDisposeScope())
            {
                long[] targetShape = features[0].shape;
                long targetH = targetShape[^2];
                long targetW = targetShape[^1];

                List<torch.Tensor> tensors = new List<torch.Tensor>(features.Length);
                tensors.Add(features[0]);

                for (int i = 1; i < features.Length; i++)
                {
                    torch.Tensor layer = features[i];
                    long[] shape = layer.shape;
                    if (shape[^2] != targetH || shape[^1] != targetW)
                    {
                        layer = torch.nn.functional.interpolate(layer, size: [targetH, targetW], mode: torch.InterpolationMode.Nearest);
                    }
                    tensors.Add(layer);
                }

                torch.Tensor result = torch.cat(tensors, 1);
                return result.MoveToOuterDisposeScope();

            }
        }

        private static torch.Tensor ComputeDistance(torch.Tensor embedding, torch.Tensor mean, torch.Tensor covariance)
        {
            using (torch.no_grad())
            using (torch.NewDisposeScope())
            {
                long batch = embedding.shape[0];
                long channel = embedding.shape[1];
                long height = embedding.shape[2];
                long width = embedding.shape[3];

                torch.Tensor embedding_reshaped = embedding.reshape(batch, channel, height * width);
                torch.Tensor delta = (embedding_reshaped - mean).permute(2, 0, 1);
                torch.Tensor distances = (torch.matmul(delta, covariance) * delta).sum(2).permute(1, 0);
                distances = distances.reshape(batch, 1, height, width);
                distances = distances.clamp(0).sqrt();
                return distances.MoveToOuterDisposeScope();
            }
        }

        private static torch.Tensor UpSample(torch.Tensor distance, long height = 224, long width = 224)
        {
            return torch.nn.functional.interpolate(distance, size: [height, width], mode: torch.InterpolationMode.Bilinear, align_corners: false);
        }

        private static torch.Tensor SmoothAnomalyMap(torch.Tensor anomalyMap, int sigma = 4)
        {
            int kernelSize = 2 * (int)(4.0 * sigma + 0.5) + 1;
            return torchvision.transforms.functional.gaussian_blur(anomalyMap, kernelSize, sigma);
        }

        public static torch.Tensor ComputeAnomalyMapInternal(torch.Tensor embedding, torch.Tensor mean, torch.Tensor covariance)
        {
            using (torch.no_grad())
            using (torch.NewDisposeScope())
            {
                torch.Tensor scoreMap = ComputeDistance(embedding, mean, covariance);
                long height = scoreMap.shape[^2] * 4;
                long width = scoreMap.shape[^1] * 4;
                torch.Tensor upSampledScoreMap = UpSample(scoreMap, height, width);
                torch.Tensor smoothedAnomalyMap = SmoothAnomalyMap(upSampledScoreMap);
                return smoothedAnomalyMap.MoveToOuterDisposeScope();
            }
        }

        public static torch.Tensor ComputeAnomalyMap(List<torch.Tensor>[] outputs, torch.Tensor mean, torch.Tensor covariance, torch.Tensor idx)
        {
            using (torch.no_grad())
            using (torch.NewDisposeScope())
            {
                torch.Tensor embedding = GetEmbedding(outputs);
                torch.Tensor embeddingVectors = torch.index_select(embedding, 1, idx);
                return ComputeAnomalyMapInternal(embeddingVectors, mean, covariance).MoveToOuterDisposeScope();
            }
        }
    }
}
