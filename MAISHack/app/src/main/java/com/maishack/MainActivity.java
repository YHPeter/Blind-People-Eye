package com.maishack;

import static com.maishack.network.UploadFile.upload;

import android.content.Context;
import android.content.Intent;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.Message;
import android.os.VibrationEffect;
import android.os.Vibrator;
import android.provider.MediaStore;
import android.speech.tts.TextToSpeech;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.result.ActivityResult;
import androidx.activity.result.ActivityResultCallback;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.FileProvider;

import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

public class MainActivity extends AppCompatActivity  {

    TextView txtView;
    Button takePhoto;
    ActivityResultLauncher activityResultLauncher;
    File file;
    File pastFile;
    String text;
    Thread curThread;
    TextToSpeech tts;
    private Handler handler = new MyHandler();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        final Vibrator vibrator = (Vibrator) getSystemService(Context.VIBRATOR_SERVICE);
        txtView = findViewById(R.id.textView);
        takePhoto = findViewById(R.id.button);




        activityResultLauncher = registerForActivityResult(new ActivityResultContracts.StartActivityForResult(), new ActivityResultCallback<ActivityResult>() {
            @Override
            public void onActivityResult(ActivityResult result) {
                if (pastFile != file) {
                    curThread = new Thread(new MyRunnable());
                    curThread.start();
                };
            }
        });

        tts = new TextToSpeech(this, new TextToSpeech.OnInitListener() {
            @Override
            public void onInit(int status) {
                if (status == TextToSpeech.SUCCESS) {
                    tts.setLanguage(Locale.CANADA);
                }
                else {
                    Log.i("TTS","Intialization failed");
                }
            }
        });
        tts.setSpeechRate(0.75f);

        takePhoto.setOnClickListener(new View.OnClickListener() {
            @RequiresApi(api = Build.VERSION_CODES.O)
            @Override
            public void onClick(View v) {
                pastFile = file;
                Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                try {
                    if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.Q) {
                        vibrator.cancel();
                        vibrator.vibrate(VibrationEffect.createPredefined(VibrationEffect.EFFECT_DOUBLE_CLICK));
                    }
                    takePhoto.setVisibility(View.GONE);
                    photoIntent(takePictureIntent);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        });
    }

    String currentPhotoPath;

    private File createImageFile() throws IOException {
        // Create an image file name
        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        String imageFileName = "JPEG_" + timeStamp + "_";
        File storageDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES);
        File image = File.createTempFile(
                imageFileName,  /* prefix */
                ".jpg",         /* suffix */
                storageDir      /* directory */
        );

        // Save a file: path for use with ACTION_VIEW intents
        currentPhotoPath = image.getAbsolutePath();
        return image;
    }

    private void photoIntent(Intent takePictureIntent) throws Exception {
        // Create the File where the photo should go
        File photoFile = null;
        try {
            photoFile = createImageFile();
        } catch (IOException ex) {
            // Error occurred while creating the File
            Log.i("Error", "Fail to create file");
        }
        // Continue only if the File was successfully created
        if (photoFile != null) {
            Uri photoURI = FileProvider.getUriForFile(this,
                    BuildConfig.APPLICATION_ID + ".provider", photoFile);


            takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoURI);
            if (takePictureIntent.resolveActivity(getPackageManager()) != null) {
                file = photoFile;

                //startActivity(takePictureIntent);
                activityResultLauncher.launch(takePictureIntent);
            } else {
                Toast.makeText(MainActivity.this, "There is no app that support this action",
                        Toast.LENGTH_SHORT).show();
            }
        }
    }

    private class MyRunnable implements Runnable {
        @Override
        public void run() {
            Message msg = Message.obtain();
            msg.what = 1;
            handler.sendMessage(msg);
            try {
                text = upload(file);
                Log.i("thread", text);
                tts.speak(text, TextToSpeech.QUEUE_FLUSH, new Bundle(), "");

                msg = Message.obtain();
                msg.what = 2;
                handler.sendMessage(msg);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    public class MyHandler extends Handler {
        @RequiresApi(api = Build.VERSION_CODES.O)
        @Override
        public void handleMessage(Message msg) {
            switch (msg.what) {
                case 1:
                    txtView.setText("processing");
                    takePhoto.setVisibility(View.GONE);
                    break;
                case 2:
                    txtView.setText("available");
                    takePhoto.setVisibility(View.VISIBLE);
                    break;
            }
        }
    }

}

