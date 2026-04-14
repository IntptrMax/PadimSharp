# PadimSharp – Patch Distribution Modeling for Anomaly Detection in C#

**PadimSharp** brings the power of **Padim** (Patch Distribution Modeling) to the C# ecosystem.  
Now you can **train and detect anomalies** entirely in C# – no Python needed!

🎯 Perfect for industrial inspection, quality control, and visual anomaly detection tasks.

---

## ✨ Why PadimSharp?

- **Pure C# implementation** – Train and run inference seamlessly in .NET  
- **No external dependencies** like Python or PyTorch  
- **Tested on MVTec** – Proven performance on real-world anomaly detection benchmarks  
- **Built-in localization** – See exactly where anomalies occur

---

## 🚀 Quick Start

```csharp
// Train on normal images
Config config = new Config();
BaseModel model = new BaseModel(config);
model.Train();

// Detect anomalies on new images
(bool predictGood, torch.Tensor image) = model.Predict(predictImagePath);
```

---

## 📦 Features

- ✅ Train on custom datasets with normal samples only  
- ✅ Fast inference suitable for real-time applications  
- ✅ Pixel‑level anomaly localization  
- ✅ Works with standard .NET image libraries (TorchSharp, etc.)

---

## 🧪 Tested On

We've validated PadimSharp on the **MVTec Anomaly Detection** dataset – achieving reliable detection and localization across multiple object and texture categories.

---

## 📸 Example Result

Here's a real detection example from our MVTec tests:

| Original Image | Predicted Anomaly Mask |
|----------------|------------------------|
| ![Original](https://github.com/user-attachments/assets/dda1aef9-9940-4952-b8bf-4d5f3907ec45) | ![Mask](https://github.com/user-attachments/assets/e00c6435-d28e-4773-ba8c-0a72d7d898d5) |

The model successfully identifies the anomalous region with pixel-level precision.

---

## 🤝 Contributing

We welcome contributions! Feel free to open issues or PRs to improve performance, documentation, or compatibility.

---

**Say goodbye to Python‑based anomaly detection pipelines. Hello, PadimSharp.** 🚀
