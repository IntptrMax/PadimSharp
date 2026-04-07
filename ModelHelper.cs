using TorchSharp;
using static TorchSharp.torch;

namespace Padim
{
    internal class ModelHelper
    {
        public static Tensor GetEmbedding(List<Tensor>[] outputs)
        {
            using (no_grad())
            using (NewDisposeScope())
            {
                Tensor t1 = torch.concat(outputs[0]);
                Tensor t2 = torch.concat(outputs[1]);
                Tensor t3 = torch.concat(outputs[2]);

                return EmbeddingConcat([t1, t2, t3]).MoveToOuterDisposeScope();
            }
        }

        private static Tensor EmbeddingConcat(Tensor[] features)
        {
            using (no_grad())
            using (NewDisposeScope())
            {
                long[] targetShape = features[0].shape;
                long targetH = targetShape[targetShape.Length - 2];
                long targetW = targetShape[targetShape.Length - 1];

                List<Tensor> tensors = new List<Tensor>(features.Length);
                tensors.Add(features[0]);

                for (int i = 1; i < features.Length; i++)
                {
                    Tensor layer = features[i];
                    long[] shape = layer.shape;
                    if (shape[shape.Length - 2] != targetH || shape[shape.Length - 1] != targetW)
                    {
                        layer = torch.nn.functional.interpolate(layer, size: [targetH, targetW ], mode: InterpolationMode.Nearest);
                    }
                    tensors.Add(layer);
                }

                Tensor result = torch.cat(tensors, 1);
                return result.MoveToOuterDisposeScope();

            }
        }

        private static Tensor ComputeDistance(Tensor embedding, Tensor mean, Tensor covariance)
        {
            using (no_grad())
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
            using (no_grad())
            using (NewDisposeScope())
            {
                Tensor scoreMap = ComputeDistance(embedding, mean, covariance);
                Tensor upSampledScoreMap = UpSample(scoreMap);
                Tensor smoothedAnomalyMap = SmoothAnomalyMap(upSampledScoreMap);
                return smoothedAnomalyMap.MoveToOuterDisposeScope();
            }
        }

        public static Tensor ComputeAnomalyMap(List<Tensor>[] outputs, Tensor mean, Tensor covariance, Tensor idx)
        {
            using (no_grad())
            using (NewDisposeScope())
            {
                Tensor embedding = GetEmbedding(outputs);
                Tensor embeddingVectors = torch.index_select(embedding, 1, idx);
                return ComputeAnomalyMapInternal(embeddingVectors, mean, covariance).MoveToOuterDisposeScope();
            }
        }
    }
}
