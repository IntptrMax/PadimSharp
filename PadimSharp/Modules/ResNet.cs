using TorchSharp;
using TorchSharp.Modules;

namespace Padim.Modules
{
    public class ResNet : torch.nn.Module<torch.Tensor, (torch.Tensor, torch.Tensor, torch.Tensor)>
    {
        public delegate torch.nn.Module<torch.Tensor, torch.Tensor> BlockFunc(int inplanes, int planes, int stride = 1, torch.nn.Module<torch.Tensor, torch.Tensor>? downsample = null, int groups = 1, int base_width = 64, int dilation = 1, Func<int, torch.nn.Module<torch.Tensor, torch.Tensor>>? norm_layer = null);

        private class BasicBlock : torch.nn.Module<torch.Tensor, torch.Tensor>
        {
            public static int expansion = 1;

            private readonly torch.nn.Module<torch.Tensor, torch.Tensor> conv1;

            private readonly torch.nn.Module<torch.Tensor, torch.Tensor> bn1;

            private readonly torch.nn.Module<torch.Tensor, torch.Tensor> conv2;

            internal readonly torch.nn.Module<torch.Tensor, torch.Tensor> bn2;

            private readonly torch.nn.Module<torch.Tensor, torch.Tensor> relu1;

            private readonly torch.nn.Module<torch.Tensor, torch.Tensor>? downsample;

            public BasicBlock(int in_planes, int planes, int stride, torch.nn.Module<torch.Tensor, torch.Tensor>? downsample = null, int groups = 1, int base_width = 64, int dilation = 1, Func<int, torch.nn.Module<torch.Tensor, torch.Tensor>>? norm_layer = null)
                : base("BasicBlock")
            {
                if (groups != 1 || base_width != 64)
                {
                    throw new ArgumentException("BasicBlock only supports groups=1 and base_width=64");
                }

                if (dilation > 1)
                {
                    throw new NotImplementedException("dilation > 1 not supported in BasicBlock");
                }

                if (norm_layer == null)
                {
                    norm_layer = (int planes) => torch.nn.BatchNorm2d(planes);
                }

                conv1 = torch.nn.Conv2d(in_planes, planes, 3L, stride, 1L, 1L, PaddingModes.Zeros, 1L, bias: false);
                bn1 = norm_layer(planes);
                relu1 = torch.nn.ReLU(inplace: true);
                conv2 = torch.nn.Conv2d(planes, planes, 3L, 1L, 1L, 1L, PaddingModes.Zeros, 1L, bias: false);
                bn2 = norm_layer(planes);
                this.downsample = downsample;
                RegisterComponents();
            }

            public override torch.Tensor forward(torch.Tensor input)
            {
                torch.Tensor target = input;
                torch.Tensor input2 = relu1.call(bn1.call(conv1.call(input)));
                input2 = bn2.call(conv2.call(input2));
                if (downsample != null)
                {
                    target = downsample.call(input);
                }

                return input2.add_(target).relu_();
            }

            protected override void Dispose(bool disposing)
            {
                if (disposing)
                {
                    conv1.Dispose();
                    bn1.Dispose();
                    conv2.Dispose();
                    bn2.Dispose();
                    relu1.Dispose();
                    downsample?.Dispose();
                }

                base.Dispose(disposing);
            }
        }

        private class Bottleneck : torch.nn.Module<torch.Tensor, torch.Tensor>
        {
            public static int expansion = 4;

            private readonly torch.nn.Module<torch.Tensor, torch.Tensor> conv1;

            private readonly torch.nn.Module<torch.Tensor, torch.Tensor> bn1;

            private readonly torch.nn.Module<torch.Tensor, torch.Tensor> conv2;

            private readonly torch.nn.Module<torch.Tensor, torch.Tensor> bn2;

            private readonly torch.nn.Module<torch.Tensor, torch.Tensor> conv3;

            internal readonly torch.nn.Module<torch.Tensor, torch.Tensor> bn3;

            private readonly torch.nn.Module<torch.Tensor, torch.Tensor> relu1;

            private readonly torch.nn.Module<torch.Tensor, torch.Tensor> relu2;

            private readonly torch.nn.Module<torch.Tensor, torch.Tensor>? downsample;

            public Bottleneck(int in_planes, int planes, int stride, torch.nn.Module<torch.Tensor, torch.Tensor>? downsample = null, int groups = 1, int base_width = 64, int dilation = 1, Func<int, torch.nn.Module<torch.Tensor, torch.Tensor>>? norm_layer = null)
                : base("Bottleneck")
            {
                if (norm_layer == null)
                {
                    norm_layer = (int planes) => torch.nn.BatchNorm2d(planes);
                }

                int num = (int)((double)planes * ((double)base_width / 64.0)) * groups;
                conv1 = torch.nn.Conv2d(in_planes, num, 1L, 1L, 0L, 1L, PaddingModes.Zeros, 1L, bias: false);
                bn1 = norm_layer(num);
                relu1 = torch.nn.ReLU(inplace: true);
                long in_channels = num;
                long out_channels = num;
                long kernel_size = 3L;
                long stride2 = stride;
                long groups2 = groups;
                conv2 = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride2, dilation, dilation, PaddingModes.Zeros, groups2, bias: false);
                bn2 = norm_layer(num);
                relu2 = torch.nn.ReLU(inplace: true);
                conv3 = torch.nn.Conv2d(num, expansion * planes, 1L, 1L, 0L, 1L, PaddingModes.Zeros, 1L, bias: false);
                bn3 = norm_layer(expansion * planes);
                this.downsample = downsample;
                RegisterComponents();
            }

            public override torch.Tensor forward(torch.Tensor input)
            {
                torch.Tensor target = input;
                torch.Tensor input2 = relu1.call(bn1.call(conv1.call(input)));
                input2 = relu2.call(bn2.call(conv2.call(input2)));
                input2 = bn3.call(conv3.call(input2));
                if (downsample != null)
                {
                    target = downsample.call(input);
                }

                return input2.add_(target).relu_();
            }

            protected override void Dispose(bool disposing)
            {
                if (disposing)
                {
                    conv1.Dispose();
                    bn1.Dispose();
                    conv2.Dispose();
                    conv3.Dispose();
                    bn2.Dispose();
                    bn3.Dispose();
                    relu1.Dispose();
                    relu2.Dispose();
                    downsample?.Dispose();
                }

                base.Dispose(disposing);
            }
        }

        private readonly torch.nn.Module<torch.Tensor, torch.Tensor> conv1;

        private readonly torch.nn.Module<torch.Tensor, torch.Tensor> bn1;

        private readonly torch.nn.Module<torch.Tensor, torch.Tensor> relu;

        private readonly torch.nn.Module<torch.Tensor, torch.Tensor> maxpool;

        private readonly Sequential layer1 = torch.nn.Sequential();

        private readonly Sequential layer2 = torch.nn.Sequential();

        private readonly Sequential layer3 = torch.nn.Sequential();

        private readonly Func<int, torch.nn.Module<torch.Tensor, torch.Tensor>> norm_layer;

        private int in_planes = 64;

        private int dilation;

        private int groups;

        private int base_width;

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                conv1.Dispose();
                bn1.Dispose();
                relu.Dispose();
                maxpool.Dispose();
                layer1.Dispose();
                layer2.Dispose();
                layer3.Dispose();
            }

            base.Dispose(disposing);
        }

        public static ResNet ResNet18(int numClasses = 1000, bool zero_init_residual = false, int groups = 1, int width_per_group = 64, (bool, bool, bool)? replace_stride_with_dilation = null, Func<int, torch.nn.Module<torch.Tensor, torch.Tensor>>? norm_layer = null, string? weights_file = null, bool skipfc = true, torch.Device? device = null)
        {
            return new ResNet("ResNet18", (int in_planes, int planes, int stride, torch.nn.Module<torch.Tensor, torch.Tensor>? downsample, int groups, int base_width, int dilation, Func<int, torch.nn.Module<torch.Tensor, torch.Tensor>>? norm_layer) => new BasicBlock(in_planes, planes, stride, downsample, groups, base_width, dilation, norm_layer), BasicBlock.expansion, new int[4] { 2, 2, 2, 2 }, numClasses, zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer, weights_file, skipfc, device);
        }

        public static ResNet ResNet34(int numClasses = 1000, bool zero_init_residual = false, int groups = 1, int width_per_group = 64, (bool, bool, bool)? replace_stride_with_dilation = null, Func<int, torch.nn.Module<torch.Tensor, torch.Tensor>>? norm_layer = null, string? weights_file = null, bool skipfc = true, torch.Device? device = null)
        {
            return new ResNet("ResNet34", (int in_planes, int planes, int stride, torch.nn.Module<torch.Tensor, torch.Tensor>? downsample, int groups, int base_width, int dilation, Func<int, torch.nn.Module<torch.Tensor, torch.Tensor>>? norm_layer) => new BasicBlock(in_planes, planes, stride, downsample, groups, base_width, dilation, norm_layer), BasicBlock.expansion, new int[4] { 3, 4, 6, 3 }, numClasses, zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer, weights_file, skipfc, device);
        }

        public static ResNet ResNet50(int numClasses = 1000, bool zero_init_residual = false, int groups = 1, int width_per_group = 64, (bool, bool, bool)? replace_stride_with_dilation = null, Func<int, torch.nn.Module<torch.Tensor, torch.Tensor>>? norm_layer = null, string? weights_file = null, bool skipfc = true, torch.Device? device = null)
        {
            return new ResNet("ResNet50", (int in_planes, int planes, int stride, torch.nn.Module<torch.Tensor, torch.Tensor>? downsample, int groups, int base_width, int dilation, Func<int, torch.nn.Module<torch.Tensor, torch.Tensor>>? norm_layer) => new Bottleneck(in_planes, planes, stride, downsample, groups, base_width, dilation, norm_layer), Bottleneck.expansion, new int[4] { 3, 4, 6, 3 }, numClasses, zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer, weights_file, skipfc, device);
        }

        public static ResNet ResNet101(int numClasses = 1000, bool zero_init_residual = false, int groups = 1, int width_per_group = 64, (bool, bool, bool)? replace_stride_with_dilation = null, Func<int, torch.nn.Module<torch.Tensor, torch.Tensor>>? norm_layer = null, string? weights_file = null, bool skipfc = true, torch.Device? device = null)
        {
            return new ResNet("ResNet101", (int in_planes, int planes, int stride, torch.nn.Module<torch.Tensor, torch.Tensor>? downsample, int groups, int base_width, int dilation, Func<int, torch.nn.Module<torch.Tensor, torch.Tensor>>? norm_layer) => new Bottleneck(in_planes, planes, stride, downsample, groups, base_width, dilation, norm_layer), Bottleneck.expansion, new int[4] { 3, 4, 23, 3 }, numClasses, zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer, weights_file, skipfc, device);
        }

        public static ResNet ResNet152(int numClasses = 1000, bool zero_init_residual = false, int groups = 1, int width_per_group = 64, (bool, bool, bool)? replace_stride_with_dilation = null, Func<int, torch.nn.Module<torch.Tensor, torch.Tensor>>? norm_layer = null, string? weights_file = null, bool skipfc = true, torch.Device? device = null)
        {
            return new ResNet("ResNet152", (int in_planes, int planes, int stride, torch.nn.Module<torch.Tensor, torch.Tensor>? downsample, int groups, int base_width, int dilation, Func<int, torch.nn.Module<torch.Tensor, torch.Tensor>>? norm_layer) => new Bottleneck(in_planes, planes, stride, downsample, groups, base_width, dilation, norm_layer), Bottleneck.expansion, new int[4] { 3, 8, 36, 3 }, numClasses, zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer, weights_file, skipfc, device);
        }

        public static ResNet WideResNet50_2(int num_classes = 1000, bool zero_init_residual = false, int groups = 1, (bool, bool, bool)? replace_stride_with_dilation = null, Func<int, torch.nn.Module<torch.Tensor, torch.Tensor>>? norm_layer = null, string? weights_file = null, bool skipfc = true, torch.Device? device = null)
        {
            return ResNet.ResNet50(num_classes, zero_init_residual, groups, 128, replace_stride_with_dilation, norm_layer, weights_file, skipfc, device);
        }

        public static ResNet WideResNet101_2(int num_classes = 1000, bool zero_init_residual = false, int groups = 1, (bool, bool, bool)? replace_stride_with_dilation = null, Func<int, torch.nn.Module<torch.Tensor, torch.Tensor>>? norm_layer = null, string? weights_file = null, bool skipfc = true, torch.Device? device = null)
        {
            return ResNet.ResNet101(num_classes, zero_init_residual, groups, 128, replace_stride_with_dilation, norm_layer, weights_file, skipfc, device);
        }

        public ResNet(string name, BlockFunc block, int expansion, IList<int> layers, int numClasses = 1000, bool zero_init_residual = false, int groups = 1, int width_per_group = 64, (bool, bool, bool)? replace_stride_with_dilation = null, Func<int, torch.nn.Module<torch.Tensor, torch.Tensor>>? norm_layer = null, string? weights_file = null, bool skipfc = true, torch.Device? device = null)
            : base(name)
        {
            norm_layer = (Func<int, torch.nn.Module<torch.Tensor, torch.Tensor>>?)((norm_layer != null) ? ((Func<int, torch.nn.Module>)norm_layer) : ((Func<int, torch.nn.Module>)(Func<int, torch.nn.Module<torch.Tensor, torch.Tensor>>)((int planes) => torch.nn.BatchNorm2d(planes))));
            this.norm_layer = norm_layer!;
            in_planes = 64;
            dilation = 1;
            this.groups = groups;
            base_width = width_per_group;
            (bool, bool, bool) tuple = (replace_stride_with_dilation.HasValue ? replace_stride_with_dilation.Value : (false, false, false));
            conv1 = torch.nn.Conv2d(3L, in_planes, 7L, 2L, 3L, 1L, PaddingModes.Zeros, 1L, bias: false);
            bn1 = norm_layer!(in_planes);
            relu = torch.nn.ReLU(inplace: true);
            maxpool = torch.nn.MaxPool2d(3L, 2L, 1L);
            MakeLayer(layer1, block, expansion, 64, layers[0], 1);
            MakeLayer(layer2, block, expansion, 128, layers[1], 2, tuple.Item1);
            MakeLayer(layer3, block, expansion, 256, layers[2], 2, tuple.Item2);
            RegisterComponents();
            if (string.IsNullOrEmpty(weights_file))
            {
                foreach (var item3 in named_modules())
                {
                    torch.nn.Module item = item3.module;
                    if (!(item is Conv2d conv2d))
                    {
                        if (!(item is BatchNorm2d batchNorm2d))
                        {
                            if (item is GroupNorm groupNorm)
                            {
                                torch.nn.init.constant_(groupNorm.weight, 1);
                                torch.nn.init.constant_(groupNorm.bias, 0);
                            }
                        }
                        else
                        {
                            torch.nn.init.constant_(batchNorm2d.weight, 1);
                            torch.nn.init.constant_(batchNorm2d.bias, 0);
                        }
                    }
                    else
                    {
                        torch.nn.init.kaiming_normal_(conv2d.weight, 0.0, torch.nn.init.FanInOut.FanOut, torch.nn.init.NonlinearityType.ReLU);
                    }
                }

                if (zero_init_residual)
                {
                    foreach (var item4 in named_modules())
                    {
                        torch.nn.Module item2 = item4.module;
                        if (!(item2 is BasicBlock basicBlock))
                        {
                            if (item2 is Bottleneck { bn3: BatchNorm2d bn })
                            {
                                torch.nn.init.constant_(bn.weight, 0);
                            }
                        }
                        else if (basicBlock.bn2 is BatchNorm2d batchNorm2d2)
                        {
                            torch.nn.init.constant_(batchNorm2d2.weight, 0);
                        }
                    }
                }
            }
            else
            {
                load(weights_file, strict: true, (!skipfc) ? null : new string[2] { "fc.weight", "fc.bias" });
            }

            if (device != null && device.type != 0)
            {
                this.to(device);
            }
        }

        private void MakeLayer(Sequential modules, BlockFunc block, int expansion, int planes, int blocks, int stride, bool dilate = false)
        {
            Sequential? downsample = null;
            int num = dilation;
            if (dilate)
            {
                dilation *= stride;
                stride = 1;
            }

            if (stride != 1 || in_planes != planes * expansion)
            {
                downsample = torch.nn.Sequential(torch.nn.Conv2d(in_planes, planes * expansion, 1L, stride, 0L, 1L, PaddingModes.Zeros, 1L, bias: false), norm_layer(planes * expansion));
            }

            modules.append(block(in_planes, planes, stride, downsample, groups, base_width, num, norm_layer));
            in_planes = planes * expansion;
            for (int i = 1; i < blocks; i++)
            {
                modules.append(block(in_planes, planes, 1, null, groups, base_width, dilation, norm_layer));
            }
        }

        public override (torch.Tensor, torch.Tensor, torch.Tensor) forward(torch.Tensor input)
        {
            using (DisposeScope disposeScope = torch.NewDisposeScope())
            {
                torch.Tensor feat1 = relu.call(bn1.call(conv1.call(input)));
                torch.Tensor feat2 = layer1.call(maxpool.call(feat1));
                torch.Tensor feat3 = layer2.call(feat2);
                torch.Tensor feat4 = layer3.call(feat3);
                return (feat2.MoveToOuterDisposeScope(), feat3.MoveToOuterDisposeScope(), feat4.MoveToOuterDisposeScope());
            }
        }
    }
}
