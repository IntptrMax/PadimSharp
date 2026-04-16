using TorchSharp;

namespace PadimSharp.Utils
{
    internal class PrecisionRecall
    {
        private torch.Tensor yTrue;
        private torch.Tensor yScores;
        private float[] precisions;
        private float[] recalls;
        private float[] thresholds;

        public float[] Precisions => precisions;
        public float[] Recalls => recalls;

        public PrecisionRecall(torch.Tensor yTrue, torch.Tensor yScores)
        {
            this.yTrue = yTrue.to_type(torch.ScalarType.Int64);
            this.yScores = yScores.to_type(torch.ScalarType.Float32);
            Calculate();
        }

        private void Calculate()
        {
            (this.precisions, this.recalls, this.thresholds) = _precision_recall_curve_compute_single_class(yTrue, yScores);
        }

        public float GetThreshold()
        {
            if (thresholds.Length == 0)
            {
                return 0;
            }

            torch.Tensor pre = torch.tensor(precisions);
            torch.Tensor rec = torch.tensor(recalls);
            torch.Tensor thr = torch.tensor(thresholds);

            torch.Tensor valid_pre = pre[..^1];
            torch.Tensor valid_rec = rec[..^1];

            torch.Tensor f1_score = (2 * valid_pre * valid_rec) / (valid_pre + valid_rec + 1e-6f);

            int maxIdx = torch.argmax(f1_score).ToInt32();
            return thresholds[maxIdx];
        }

        private (float[] precisions, float[] recalls, float[] thresholds) _precision_recall_curve_compute_single_class(torch.Tensor yTrue, torch.Tensor yScores, int pos_label = 1)
        {
            (torch.Tensor fps, torch.Tensor tps, torch.Tensor thresholds) = BinaryClfCurve(yScores, yTrue, pos_label);

            if (tps[-1].ToSingle() == 0f)
            {
                return (new float[] { 1f, 1f }, new float[] { 0f, 0f }, new float[] { float.MaxValue });
            }

            torch.Tensor precision = tps / (tps + fps);
            torch.Tensor recall = tps / tps[-1];

            int lastInd = torch.where(tps == tps[-1])[0][0].ToInt32();

            torch.Tensor sl = torch.arange(lastInd + 1, dtype: torch.ScalarType.Int64, device: precision.device);

            torch.Tensor reversedPrecision = precision[sl].flip(0);
            torch.Tensor reversedRecall = recall[sl].flip(0);
            torch.Tensor reversedThresholds = thresholds[sl].flip(0);

            precision = torch.cat(new torch.Tensor[] { reversedPrecision, torch.ones(1, dtype: precision.dtype, device: precision.device) });
            recall = torch.cat(new torch.Tensor[] { reversedRecall, torch.zeros(1, dtype: recall.dtype, device: recall.device) });

            return (precision.data<float>().ToArray(), recall.data<float>().ToArray(), reversedThresholds.data<float>().ToArray());
        }

        private (torch.Tensor fps, torch.Tensor tps, torch.Tensor thresholds) BinaryClfCurve(torch.Tensor preds, torch.Tensor target, int posLabel = 1)
        {
            using (torch.no_grad())
            {
                if (preds.ndim > target.ndim)
                {
                    preds = preds[torch.TensorIndex.Ellipsis, 0];
                }

                torch.Tensor descScoreIndices = torch.argsort(preds, descending: true);
                preds = preds[descScoreIndices];
                target = target[descScoreIndices];

                torch.Tensor diff = preds[1..] - preds[..^1];
                torch.Tensor distinctValueIndices = torch.nonzero(diff != 0);

                distinctValueIndices = distinctValueIndices.reshape(-1);

                torch.Tensor lastIdx = torch.tensor(new long[] { target.shape[0] - 1 }, device: preds.device);
                torch.Tensor thresholdIdxs = torch.cat(new torch.Tensor[] { distinctValueIndices, lastIdx });

                target = (target == posLabel).to_type(torch.ScalarType.Int64);

                torch.Tensor tps = torch.cumsum(target, dim: 0)[thresholdIdxs];

                torch.Tensor fps = (thresholdIdxs + 1) - tps;

                return (fps, tps, preds[thresholdIdxs]);
            }
        }

        public (float precision, float recall, float f1) GetBestMetrics()
        {
            if (thresholds.Length == 0) return (0, 0, 0);

            torch.Tensor pre = torch.tensor(precisions);
            torch.Tensor rec = torch.tensor(recalls);

            torch.Tensor valid_pre = pre[..^1];
            torch.Tensor valid_rec = rec[..^1];

            torch.Tensor f1_scores = (2 * valid_pre * valid_rec) / (valid_pre + valid_rec + 1e-6f);
            int bestIdx = torch.argmax(f1_scores).ToInt32();

            float bestP = valid_pre[bestIdx].ToSingle();
            float bestR = valid_rec[bestIdx].ToSingle();
            float bestF1 = f1_scores[bestIdx].ToSingle();

            return (bestP, bestR, bestF1);
        }

        public float GetAveragePrecision()
        {
            torch.Tensor pre = torch.tensor(precisions);
            torch.Tensor rec = torch.tensor(recalls);

            float ap = 0;
            for (int i = 0; i < precisions.Length - 1; i++)
            {
                ap += (recalls[i] - recalls[i + 1]) * precisions[i];
            }
            return Math.Abs(ap);
        }
    }
}