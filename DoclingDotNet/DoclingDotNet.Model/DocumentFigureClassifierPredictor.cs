using SixLabors.ImageSharp.ColorSpaces;

namespace DoclingDotNet.Model;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Runtime.CompilerServices;

public class DocumentFigureClassifierPredictor : IDisposable
{
    private static readonly string[] Classes = new[]
    {
        "bar_chart",
        "bar_code",
        "chemistry_markush_structure",
        "chemistry_molecular_structure",
        "flow_chart",
        "icon",
        "line_chart",
        "logo",
        "map",
        "other",
        "pie_chart",
        "qr_code",
        "remote_sensing",
        "screenshot",
        "signature",
        "stamp"
    };

    private readonly string _device;
    private readonly int _numThreads;
    private readonly InferenceSession _session;

    // Match Python's normalization parameters
    private static readonly float[] ImageMean = new float[] { 0.485f, 0.456f, 0.406f };
    private static readonly float[] ImageStd = new float[] { 0.47853944f, 0.4732864f, 0.47434163f };
    private const int ImageSize = 224;

    public DocumentFigureClassifierPredictor(string artifactsPath, string device = "cpu", int numThreads = 4)
    {
        _device = device;
        _numThreads = numThreads;

        var sessionOptions = new SessionOptions();
        if (device == "cpu")
        {
            sessionOptions.InterOpNumThreads = numThreads;
            sessionOptions.IntraOpNumThreads = numThreads;
        }
        
        _session = new InferenceSession(Path.Combine(artifactsPath, "model.onnx"), sessionOptions);
    }

    public IDictionary<string, object> GetInfo()
    {
        return new Dictionary<string, object>
        {
            ["device"] = _device,
            ["num_threads"] = _numThreads,
            ["classes"] = Classes
        };
    }

    private DenseTensor<float> ProcessImage(Image<Rgb24> image)
    {
        // Resize image to 224x224
        image.Mutate(x => x.Resize(ImageSize, ImageSize));

        // Create tensor for batch size 1, 3 channels, height 224, width 224
        var tensor = new DenseTensor<float>(new[] { 1, 3, ImageSize, ImageSize });

        // Convert image to tensor with normalization
        for (int y = 0; y < ImageSize; y++)
        {
            for (int x = 0; x < ImageSize; x++)
            {
                var pixel = image[x, y];
                // Convert RGB values to float and normalize
                tensor[0, 0, y, x] = ((pixel.R / 255f) - ImageMean[0]) / ImageStd[0];
                tensor[0, 1, y, x] = ((pixel.G / 255f) - ImageMean[1]) / ImageStd[1];
                tensor[0, 2, y, x] = ((pixel.B / 255f) - ImageMean[2]) / ImageStd[2];
            }
        }

        return tensor;
    }

    // Helper method to process multiple images
    private IEnumerable<DenseTensor<float>> ProcessImages(IEnumerable<Image<Rgb24>> images)
    {
        return images.Select(img => ProcessImage(img));
    }

    public List<List<(string Label, float Probability)>> Predict(IEnumerable<Image<Rgb24>> images)
    {
        var tensorInputs = ProcessImages(images).ToList();
        var result = new List<List<(string, float)>>();

        foreach (var input in tensorInputs)
        {
            var inputMeta = _session.InputMetadata;
            var inputName = inputMeta.Keys.First();
            
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(inputName, input)
            };

            var outputs = _session.Run(inputs);
            var logits = outputs.First().AsTensor<float>();

            // Convert logits to probabilities using softmax
            var probabilities = Softmax(logits.ToArray());

            // Create predictions list with class labels and probabilities
            var predictions = new List<(string, float)>();
            for (int i = 0; i < Classes.Length; i++)
            {
                predictions.Add((Classes[i], probabilities[i]));
            }

            // Sort by probability in descending order
            predictions.Sort((a, b) => b.Item2.CompareTo(a.Item2));
            result.Add(predictions);
        }

        return result;
    }

    private float[] Softmax(float[] logits)
    {
        var maxLogit = logits.Max();
        var exp = logits.Select(x => Math.Exp(x - maxLogit)).ToArray();
        var sumExp = exp.Sum();
        return exp.Select(x => (float)(x / sumExp)).ToArray();
    }

    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    protected virtual void Dispose(bool disposing)
    {
        if (disposing)
        {
            _session?.Dispose();
        }
    }

    ~DocumentFigureClassifierPredictor()
    {
        Dispose(false);
    }
}