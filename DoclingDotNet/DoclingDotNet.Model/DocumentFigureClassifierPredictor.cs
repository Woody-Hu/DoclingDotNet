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

    private DenseTensor<float> ProcessImageBatch(IEnumerable<Image<Rgb24>> images)
    {
        var imageList = images.ToList();
        var batchSize = imageList.Count;
        
        // Create tensor for variable batch size, 3 channels, height 224, width 224
        var tensor = new DenseTensor<float>(new[] { batchSize, 3, ImageSize, ImageSize });

        for (int b = 0; b < batchSize; b++)
        {
            var image = imageList[b];
            image.Mutate(x => x.Resize(ImageSize, ImageSize));

            for (int y = 0; y < ImageSize; y++)
            {
                for (int x = 0; x < ImageSize; x++)
                {
                    var pixel = image[x, y];
                    tensor[b, 0, y, x] = ((pixel.R / 255f) - ImageMean[0]) / ImageStd[0];
                    tensor[b, 1, y, x] = ((pixel.G / 255f) - ImageMean[1]) / ImageStd[1];
                    tensor[b, 2, y, x] = ((pixel.B / 255f) - ImageMean[2]) / ImageStd[2];
                }
            }
        }

        return tensor;
    }

    public List<List<(string Label, float Probability)>> Predict(IEnumerable<Image<Rgb24>> images)
    {
        // Process all images in a single batch
        var batchTensor = ProcessImageBatch(images);
        
        var inputMeta = _session.InputMetadata;
        var inputName = inputMeta.Keys.First();
        
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(inputName, batchTensor)
        };

        // Single inference call for all images
        using var outputs = _session.Run(inputs);
        var logits = outputs.First().AsTensor<float>();
        
        var result = new List<List<(string, float)>>();
        var batchSize = logits.Dimensions[0];
        
        // Process each image's results
        for (int b = 0; b < batchSize; b++)
        {
            var imageLogits = new float[Classes.Length];
            for (int i = 0; i < Classes.Length; i++)
            {
                imageLogits[i] = logits[b, i];
            }

            var probabilities = Softmax(imageLogits);
            var predictions = new List<(string, float)>();
            
            for (int i = 0; i < Classes.Length; i++)
            {
                predictions.Add((Classes[i], probabilities[i]));
            }

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