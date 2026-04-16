using TorchSharp;

namespace PadimSharp.Modules
{
    public class Weight : torch.nn.Module
    {
        public Weight(int d = 100, int height = 224, int width = 224) : base("profile")
        {
            int sz = height * width / 16;
            this.register_buffer("cov", torch.zeros(sz, d, d));
            this.register_buffer("mean", torch.zeros(d, sz));
            this.register_buffer("image_threshold", torch.zeros(1));
            this.register_buffer("pixel_threshold", torch.zeros(1));
            this.register_buffer("idx", torch.zeros(d, torch.ScalarType.Int32));
        }

        public void Save(string path, float image_threshold, float pixel_threshold, torch.Tensor idx, torch.Tensor cov, torch.Tensor mean)
        {
            this.get_buffer("cov")!.copy_(cov);
            this.get_buffer("mean")!.copy_(mean);
            this.get_buffer("idx")!.copy_(idx);
            this.get_buffer("image_threshold")!.copy_(image_threshold);
            this.get_buffer("pixel_threshold")!.copy_(pixel_threshold);
            this.save(path);
        }

        public (float image_threshold, float pixel_threshold, torch.Tensor idx, torch.Tensor cov, torch.Tensor mean) Load(string path)
        {
            this.load(path);
            torch.Tensor idx = this.get_buffer("idx")!;
            torch.Tensor cov = this.get_buffer("cov")!;
            torch.Tensor mean = this.get_buffer("mean")!;
            float image_threshold = this.get_buffer("image_threshold")!.ToSingle();
            float pixel_threshold = this.get_buffer("pixel_threshold")!.ToSingle();
            return (image_threshold, pixel_threshold, idx, cov, mean);
        }
    }
}
