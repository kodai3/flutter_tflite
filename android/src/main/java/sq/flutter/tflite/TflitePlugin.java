package sq.flutter.tflite;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.BitmapFactory;
import android.os.SystemClock;
import android.renderscript.Allocation;
import android.renderscript.Element;
import android.renderscript.RenderScript;
import android.renderscript.ScriptIntrinsicYuvToRGB;
import android.renderscript.Type;
import android.util.Log;

import io.flutter.plugin.common.MethodCall;
import io.flutter.plugin.common.MethodChannel;
import io.flutter.plugin.common.MethodChannel.MethodCallHandler;
import io.flutter.plugin.common.MethodChannel.Result;
import io.flutter.plugin.common.PluginRegistry.Registrar;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Vector;

public class TflitePlugin implements MethodCallHandler {
  private final Registrar mRegistrar;
  private Interpreter tfLite;
  private int inputSize = 0;
  private Vector<String> labels;
  float[][] labelProb;
  private static final int BYTES_PER_CHANNEL = 4;

  public static void registerWith(Registrar registrar) {
    final MethodChannel channel = new MethodChannel(registrar.messenger(), "tflite");
    channel.setMethodCallHandler(new TflitePlugin(registrar));
  }

  private TflitePlugin(Registrar registrar) {
    this.mRegistrar = registrar;
  }

  @Override
  public void onMethodCall(MethodCall call, Result result) {
    if (call.method.equals("loadModel")) {
      try {
        String res = loadModel((HashMap) call.arguments);
        result.success(res);
      } catch (Exception e) {
        result.error("Failed to load model", e.getMessage(), e);
      }
    } else if (call.method.equals("detectObjectOnImage")) {
      try {
        List<Map<String, Object>> res = detectObjectOnImage((HashMap) call.arguments);
        result.success(res);
      } catch (Exception e) {
        result.error("Failed to run model", e.getMessage(), e);
      }
    } else if (call.method.equals("close")) {
      close();
    }
  }

  private String loadModel(HashMap args) throws IOException {
    String model = args.get("model").toString();
    AssetManager assetManager = mRegistrar.context().getAssets();
    String key = mRegistrar.lookupKeyForAsset(model);
    AssetFileDescriptor fileDescriptor = assetManager.openFd(key);
    FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
    FileChannel fileChannel = inputStream.getChannel();
    long startOffset = fileDescriptor.getStartOffset();
    long declaredLength = fileDescriptor.getDeclaredLength();
    MappedByteBuffer buffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);

    int numThreads = (int) args.get("numThreads");
    final Interpreter.Options tfliteOptions = new Interpreter.Options();
    tfliteOptions.setNumThreads(numThreads);
    tfLite = new Interpreter(buffer, tfliteOptions);

    String labels = args.get("labels").toString();
    key = mRegistrar.lookupKeyForAsset(labels);
    loadLabels(assetManager, key);

    return "success";
  }

  private void loadLabels(AssetManager assetManager, String path) {
    BufferedReader br;
    try {
      br = new BufferedReader(new InputStreamReader(assetManager.open(path)));
      String line;
      labels = new Vector<>();
      while ((line = br.readLine()) != null) {
        labels.add(line);
      }
      labelProb = new float[1][labels.size()];
      br.close();
    } catch (IOException e) {
      throw new RuntimeException("Failed to read label file", e);
    }
  }

  ByteBuffer feedInputTensorImage(String path, float mean, float std) throws IOException {
    InputStream inputStream = new FileInputStream(path.replace("file://", ""));
    Bitmap bitmapRaw = BitmapFactory.decodeStream(inputStream);

    return feedInputTensor(bitmapRaw, mean, std);
  }

  ByteBuffer feedInputTensorFrame(List<byte[]> bytesList, int imageHeight, int imageWidth, float mean, float std,
      int rotation) throws IOException {
    ByteBuffer Y = ByteBuffer.wrap(bytesList.get(0));
    ByteBuffer U = ByteBuffer.wrap(bytesList.get(1));
    ByteBuffer V = ByteBuffer.wrap(bytesList.get(2));

    int Yb = Y.remaining();
    int Ub = U.remaining();
    int Vb = V.remaining();

    byte[] data = new byte[Yb + Ub + Vb];

    Y.get(data, 0, Yb);
    V.get(data, Yb, Vb);
    U.get(data, Yb + Vb, Ub);

    Bitmap bitmapRaw = Bitmap.createBitmap(imageWidth, imageHeight, Bitmap.Config.ARGB_8888);
    Allocation bmData = renderScriptNV21ToRGBA888(mRegistrar.context(), imageWidth, imageHeight, data);
    bmData.copyTo(bitmapRaw);

    Matrix matrix = new Matrix();
    matrix.postRotate(rotation);
    bitmapRaw = Bitmap.createBitmap(bitmapRaw, 0, 0, bitmapRaw.getWidth(), bitmapRaw.getHeight(), matrix, true);

    return feedInputTensor(bitmapRaw, mean, std);
  }

  public Allocation renderScriptNV21ToRGBA888(Context context, int width, int height, byte[] nv21) {
    // https://stackoverflow.com/a/36409748
    RenderScript rs = RenderScript.create(context);
    ScriptIntrinsicYuvToRGB yuvToRgbIntrinsic = ScriptIntrinsicYuvToRGB.create(rs, Element.U8_4(rs));

    Type.Builder yuvType = new Type.Builder(rs, Element.U8(rs)).setX(nv21.length);
    Allocation in = Allocation.createTyped(rs, yuvType.create(), Allocation.USAGE_SCRIPT);

    Type.Builder rgbaType = new Type.Builder(rs, Element.RGBA_8888(rs)).setX(width).setY(height);
    Allocation out = Allocation.createTyped(rs, rgbaType.create(), Allocation.USAGE_SCRIPT);

    in.copyFrom(nv21);

    yuvToRgbIntrinsic.setInput(in);
    yuvToRgbIntrinsic.forEach(out);
    return out;
  }

  private List<Map<String, Object>> runModelOnImage(HashMap args) throws IOException {
    String path = args.get("path").toString();
    double mean = (double) (args.get("imageMean"));
    float IMAGE_MEAN = (float) mean;
    double std = (double) (args.get("imageStd"));
    float IMAGE_STD = (float) std;
    int NUM_RESULTS = (int) args.get("numResults");
    double threshold = (double) args.get("threshold");
    float THRESHOLD = (float) threshold;

    long startTime = SystemClock.uptimeMillis();
    tfLite.run(feedInputTensorImage(path, IMAGE_MEAN, IMAGE_STD), labelProb);
    Log.v("time", "Inference took " + (SystemClock.uptimeMillis() - startTime));

    return GetTopN(NUM_RESULTS, THRESHOLD);
  }

  private List<Map<String, Object>> runModelOnBinary(HashMap args) throws IOException {
    byte[] binary = (byte[]) args.get("binary");
    int NUM_RESULTS = (int) args.get("numResults");
    double threshold = (double) args.get("threshold");
    float THRESHOLD = (float) threshold;

    ByteBuffer imgData = ByteBuffer.wrap(binary);
    tfLite.run(imgData, labelProb);

    return GetTopN(NUM_RESULTS, THRESHOLD);
  }

  private List<Map<String, Object>> detectObjectOnImage(HashMap args) throws IOException {
    String path = args.get("path").toString();
    String model = args.get("model").toString();
    double mean = (double) (args.get("imageMean"));
    float IMAGE_MEAN = (float) mean;
    double std = (double) (args.get("imageStd"));
    float IMAGE_STD = (float) std;
    double threshold = (double) args.get("threshold");
    float THRESHOLD = (float) threshold;
    int NUM_RESULTS_PER_CLASS = (int) args.get("numResultsPerClass");

    ByteBuffer imgData = feedInputTensorImage(path, IMAGE_MEAN, IMAGE_STD);

    return parseSSDMobileNet(imgData, NUM_RESULTS_PER_CLASS, THRESHOLD);
  }

  private List<Map<String, Object>> detectObjectOnFrame(HashMap args) throws IOException {
    List<byte[]> bytesList = (ArrayList) args.get("bytesList");
    String model = args.get("model").toString();
    double mean = (double) (args.get("imageMean"));
    float IMAGE_MEAN = (float) mean;
    double std = (double) (args.get("imageStd"));
    float IMAGE_STD = (float) std;
    int imageHeight = (int) (args.get("imageHeight"));
    int imageWidth = (int) (args.get("imageWidth"));
    int rotation = (int) (args.get("rotation"));
    double threshold = (double) args.get("threshold");
    float THRESHOLD = (float) threshold;
    int NUM_RESULTS_PER_CLASS = (int) args.get("numResultsPerClass");

    ByteBuffer imgData = feedInputTensorFrame(bytesList, imageHeight, imageWidth, IMAGE_MEAN, IMAGE_STD, rotation);

    return parseSSDMobileNet(imgData, NUM_RESULTS_PER_CLASS, THRESHOLD);
  }

  private List<Map<String, Object>> parseSSDMobileNet(ByteBuffer imgData, int numResultsPerClass, float threshold) {
    int NUM_DETECTIONS = 10;
    float[][][] outputLocations = new float[1][NUM_DETECTIONS][4];
    float[][] outputClasses = new float[1][NUM_DETECTIONS];
    float[][] outputScores = new float[1][NUM_DETECTIONS];
    float[] numDetections = new float[1];

    Object[] inputArray = { imgData };
    Map<Integer, Object> outputMap = new HashMap<>();
    outputMap.put(0, outputLocations);
    outputMap.put(1, outputClasses);
    outputMap.put(2, outputScores);
    outputMap.put(3, numDetections);

    long startTime = SystemClock.uptimeMillis();

    tfLite.runForMultipleInputsOutputs(inputArray, outputMap);

    Log.v("time", "Inference took " + (SystemClock.uptimeMillis() - startTime));

    Map<String, Integer> counters = new HashMap<>();
    final List<Map<String, Object>> results = new ArrayList<>(NUM_DETECTIONS);

    for (int i = 0; i < NUM_DETECTIONS; ++i) {
      if (outputScores[0][i] < threshold)
        continue;

      String detectedClass = labels.get((int) outputClasses[0][i] + 1);

      if (counters.get(detectedClass) == null) {
        counters.put(detectedClass, 1);
      } else {
        int count = counters.get(detectedClass);
        if (count >= numResultsPerClass) {
          continue;
        } else {
          counters.put(detectedClass, count + 1);
        }
      }

      Map<String, Object> rect = new HashMap<>();
      float ymin = Math.max(0, outputLocations[0][i][0]);
      float xmin = Math.max(0, outputLocations[0][i][1]);
      float ymax = outputLocations[0][i][2];
      float xmax = outputLocations[0][i][3];
      rect.put("x", xmin);
      rect.put("y", ymin);
      rect.put("w", Math.min(1 - xmin, xmax - xmin));
      rect.put("h", Math.min(1 - ymin, ymax - ymin));

      Map<String, Object> result = new HashMap<>();
      result.put("rect", rect);
      result.put("confidenceInClass", outputScores[0][i]);
      result.put("detectedClass", detectedClass);

      results.add(result);
    }

    return results;
  }

  private void close() {
    tfLite.close();
    labels = null;
    labelProb = null;
  }
}
