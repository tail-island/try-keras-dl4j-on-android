package com.tail_island.consume_model;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.AsyncTask;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;

import org.datavec.image.loader.AndroidNativeImageLoader;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.preprocessors.TensorFlowCnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.shade.jackson.databind.jsontype.NamedType;

import java.io.BufferedInputStream;
import java.io.InputStream;
import java.util.Arrays;

public class MainActivity extends AppCompatActivity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        AsyncTask.execute(new Runnable() {
            @Override
            public void run() {
                try {
                    doIt();

                } catch (Exception ex) {
                    Log.e("consume-model", "exception", ex);
                }
            }
        });
    }

    private void doIt() throws Exception {
        NeuralNetConfiguration.reinitMapperWithSubtypes(Arrays.asList(new NamedType(TensorFlowCnnToFeedForwardPreProcessor.class)));

        MultiLayerNetwork model;
        try (InputStream stream = new BufferedInputStream(getResources().openRawResource(R.raw.model))) {
            model = ModelSerializer.restoreMultiLayerNetwork(stream);
        }

        Bitmap bitmap = Bitmap.createBitmap(BitmapFactory.decodeResource(getResources(), R.raw.bitmap), 0, 0, 199, 82);
        INDArray input = new AndroidNativeImageLoader(160, 48, 3).asMatrix(bitmap);

        Log.i("comsume-model", model.output(input).toString());
    }
}
