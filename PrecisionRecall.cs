using TorchSharp;
using static TorchSharp.torch;

namespace Padim
{
	internal class PrecisionRecall
	{
		private Tensor yTrue;
		private Tensor yScores;
		private float[] precisions;
		private float[] recalls;
		private float[] thresholds;

		public PrecisionRecall(Tensor yTrue, Tensor yScores)
		{
			this.yTrue = yTrue;
			this.yScores = yScores;
			Calculate();
		}

		private void Calculate()
		{
			(this.precisions, this.recalls, this.thresholds) = _precision_recall_curve_compute_single_class(yTrue, yScores);
		}

		public float GetThreshold()
		{
			Tensor pre = torch.tensor(precisions);
			Tensor rec = torch.tensor(recalls);
			Tensor thr = torch.tensor(thresholds);
			var f1_score = (2 * pre * rec) / (pre + rec + 1e-10f);
			if (thresholds.Length == 1)
			{
				return thresholds[0];
			}
			else
			{
				return thresholds[torch.argmax(f1_score).ToInt32()];
			}
		}



		private (float[] precisions, float[] recalls, float[] thresholds) _precision_recall_curve_compute_single_class(Tensor yTrue, Tensor yScores, int pos_label = 1)
		{
			var (fps, tps, thresholds) = BinaryClfCurve(yScores, yTrue, pos_label);
			var precision = tps / (tps + fps);
			var recall = tps / tps[-1];

			var lastInd = torch.where(tps == tps[-1])[0][0].ToInt32();
			int[] sl = new int[lastInd + 1];
			for (int i = 0; i < sl.Length; i++)
			{
				sl[i] = i;
			}
			var reversedPrecision = precision[sl].flip(0);
			var reversedRecall = recall[sl].flip(0);
			var reversedThresholds = thresholds[sl].flip(0);

			precision = torch.cat(new Tensor[] { reversedPrecision, torch.ones(1, dtype: precision.dtype, device: precision.device) });
			recall = torch.cat(new Tensor[] { reversedRecall, torch.zeros(1, dtype: recall.dtype, device: recall.device) });

			return (precision.data<float>().ToArray(), recall.data<float>().ToArray(), reversedThresholds.data<float>().ToArray());
		}

		private (Tensor fps, Tensor tps, Tensor thresholds) BinaryClfCurve(Tensor preds, Tensor target, int posLabel = 1)
		{
			using (torch.no_grad())
			{
				if (preds.ndim > target.ndim)
				{
					preds = preds[TensorIndex.Ellipsis, 0];
				}

				var descScoreIndices = torch.argsort(preds, descending: true);
				preds = preds[descScoreIndices];
				target = target[descScoreIndices];

				Tensor weight = torch.tensor(1.0f);

				var distinctValueIndices = torch.nonzero(preds[1..] - preds[..^1]).squeeze();
				var thresholdIdxs = torch.cat(new Tensor[] { distinctValueIndices, torch.tensor(new long[] { target.shape[0] - 1 }, device: preds.device) });

				target = (target == posLabel).to_type(ScalarType.Int64);

				var tps = torch.cumsum(target * weight, dim: 0)[thresholdIdxs];

				Tensor fps = 1 + thresholdIdxs - tps;
				return (fps, tps, preds[thresholdIdxs]);
			}
		}
	}
}
