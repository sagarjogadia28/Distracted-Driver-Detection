package com.aws.sagarjogadia.aws;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.hardware.Camera;
import android.media.MediaActionSound;
import android.media.MediaPlayer;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.util.Base64;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.FrameLayout;
import android.widget.ImageButton;
import android.widget.TextView;

import com.amazonaws.mobileconnectors.s3.transferutility.TransferUtility;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Date;
import java.util.List;
import java.util.Timer;
import java.util.TimerTask;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;

public class CameraActivity extends AppCompatActivity {

    private static final String TAG = "tag";
    private Camera mCamera = null;
    TransferUtility transferUtility;
    private static int counter = 0;
    private static int distractedCounter = 0;
    private static int distractionType = -1;
    private static int INTERVAL = 2000;

    private static final String MY_BUCKET = "2aws";
    private static final String OBJECT_KEY = "Images";
    private Bitmap bitmap;
    String[] distractedClass = {"Safe Driving", "Texting - right", "Talking on the phone - right", "Texting - left", "Talking on the phone - left", " Operating the radio", "Drinking", "Reaching behind", "Hair and makeup", "Looking around"};

    AwsApi awsApi;
    AwsModel awsModel;
    TextView textView;
    int x=800,y=600;

    protected void onCreate(Bundle savedInstanceState) {
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().getDecorView().setSystemUiVisibility(
                View.SYSTEM_UI_FLAG_LAYOUT_STABLE
                        | View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
                        | View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
                        | View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                        | View.SYSTEM_UI_FLAG_FULLSCREEN
                        | View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY);
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_camera);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

//        x = getIntent().getIntExtra("Xvalue",800);
//        y = getIntent().getIntExtra("Yvalue",600);

        textView = (TextView) findViewById(R.id.textView);

        String BASE_URL = "http://34.202.19.5:8000/";
        Retrofit retrofit = new Retrofit.Builder()
                .baseUrl(BASE_URL)
                .addConverterFactory(GsonConverterFactory.create())
                .build();

        awsApi = retrofit.create(AwsApi.class);

        // Initialize the Amazon Cognito credentials provider
//        CognitoCachingCredentialsProvider credentialsProvider = new CognitoCachingCredentialsProvider(
//                getApplicationContext(),
//                "us-east-1:930817da-025f-4fcd-8b9c-f6859cde85fd", // Identity Pool ID
//                Regions.US_EAST_1 // Region
//        );

        // pass Amazon Cognito credentials provider to the S3 client constructor
//        AmazonS3 s3 = new AmazonS3Client(credentialsProvider);

        //pass the client to the TransferUtility constructor along with the application context:
//        transferUtility = new TransferUtility(s3, getApplicationContext());

        try {
            Camera.CameraInfo info = new Camera.CameraInfo();
            int count = Camera.getNumberOfCameras();

            for (int i = 0; i < count; i++) {
                Camera.getCameraInfo(i, info);
                if (info.facing == Camera.CameraInfo.CAMERA_FACING_BACK) {
                    try {
                        mCamera = Camera.open(i);
                    } catch (RuntimeException e) {
                        // Handle
                    }
                }
            }
        } catch (Exception e) {
            Log.d("ERROR", "Failed to get camera: " + e.getMessage());
        }

        //Change picture size
        Camera.Parameters params = mCamera.getParameters();
        // Check what resolutions are supported by your camera
        List<Camera.Size> sizes = params.getSupportedPictureSizes();

        // Iterate through all available resolutions and choose one.
        // The chosen resolution will be stored in mSize.
//        Camera.Size mSize = null;
//        for (Camera.Size size : sizes) {
//            mSize = size;
////            Log.d(TAG, "onCreate: " + mSize.width + " " + mSize.height);
//        }

        params.setPictureSize(x, y);
        mCamera.setParameters(params);

        if (mCamera != null) {
            CameraView mCameraView = new CameraView(this, mCamera);
            FrameLayout camera_view = (FrameLayout) findViewById(R.id.camera_view);
            camera_view.addView(mCameraView);//add the SurfaceView to the layout
        }

        //button to close the application
        ImageButton imgClose = (ImageButton) findViewById(R.id.imgClose);
        imgClose.setOnClickListener(new View.OnClickListener()

        {
            @Override
            public void onClick(View view) {
                System.exit(0);
            }
        });

        //Take pictures at regular intervals
        Timer t = new Timer();
        TimerTask task = new TimerTask() {

            @Override
            public void run() {
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        MediaActionSound sound = new MediaActionSound();
                        sound.play(MediaActionSound.SHUTTER_CLICK);
                        mCamera.takePicture(null, null, pictureCallback);
                    }
                });
            }
        };
        t.scheduleAtFixedRate(task, 0, INTERVAL);

    }

    public static byte[] convertImageToBase64(Bitmap image) {
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
        image.compress(Bitmap.CompressFormat.JPEG, 100, byteArrayOutputStream); //bm is the bitmap object
        byte[] byteArrayImage = byteArrayOutputStream.toByteArray();
        return Base64.encode(byteArrayImage, 0);
    }

    Camera.PictureCallback pictureCallback = new Camera.PictureCallback() {

        @Override
        public void onPictureTaken(final byte[] data, Camera camera) {
            // TODO Auto-generated method stub
            File file = new File(android.os.Environment.getExternalStorageDirectory(), File.separator + "AWSData");
            if (!file.exists()) {
                file.mkdirs();
            }
            FileOutputStream imageFileOS;
            try {
                file = new File(android.os.Environment.getExternalStorageDirectory(), File.separator + "AWSData" + File.separator + "face" + counter + ".jpg");
                if (!file.exists()) {
                    file.createNewFile();
                }
                imageFileOS = new FileOutputStream(file, false);
                imageFileOS.write(data);
                final Date date = new Date(file.lastModified());
//                Log.d(TAG, "File store Time => " + date.getTime());

                bitmap = BitmapFactory.decodeFile(file.getAbsolutePath());
                byte[] base64File = convertImageToBase64(bitmap);
//                imageFileOS = new FileOutputStream(file1, false);
//                imageFileOS.write(base64File);

                awsModel = new AwsModel();
                awsModel.setDirection("right");
                awsModel.setFileName(file.getName());
                awsModel.setImageData(new String(base64File));
                Call<AwsResponse> call = awsApi.sendData(awsModel);
                call.enqueue(new Callback<AwsResponse>() {
                    @Override
                    public void onResponse(Call<AwsResponse> call, Response<AwsResponse> response) {
                        Long tsLong = System.currentTimeMillis() / 1000;
//                        Log.d(TAG, "Current Time => "+tsLong);
                        long diff = tsLong - date.getTime();
//                        Log.d(TAG, "Difference  => " + diff);
                        Log.d(TAG, "onResponse: " + response.code());
                        AwsResponse awsResponse = response.body();
                        if(awsResponse != null){
                            Log.d(TAG, "onResponse: Message ->" + awsResponse.getMessage());
                            Log.d(TAG, "onResponse: Filename ->" + awsResponse.getFileName());
                            Log.d(TAG, "onResponse: DistractedType ->" + awsResponse.getDistractedType());
                            int type = awsResponse.getDistractedType();
                            Log.d(TAG, "onResponse: Distracted=>" + distractedClass[type]);


                            if (type == 9) {
                                distractedCounter++;
                            }
                            if (distractedCounter > 2 && type != 0) {
                                textView.setText(distractedClass[type]);
                                textView.setTextColor(Color.WHITE);
//                            Toast.makeText(CameraActivity.this, "Type: " + distractedClass[type], Toast.LENGTH_SHORT).show();
                                MediaPlayer mediaPlayer = MediaPlayer.create(getApplicationContext(), R.raw.alert);
                                mediaPlayer.setLooping(false);
                                mediaPlayer.start();
                                distractedCounter = 0;
                            }
                            if (type != 9 && type != 0) {
//                            Toast.makeText(CameraActivity.this, "Type: " + distractedClass[type], Toast.LENGTH_LONG).show();
                                textView.setText(distractedClass[type]);
                                textView.setTextColor(Color.WHITE);
                                MediaPlayer mediaPlayer = MediaPlayer.create(getApplicationContext(), R.raw.alert);
                                mediaPlayer.setLooping(false);
                                mediaPlayer.start();
                            }
                        }
                    }

                    @Override
                    public void onFailure(Call<AwsResponse> call, Throwable t) {

                    }
                });

                //upload to the bucket
//                TransferObserver observer = transferUtility.upload(
//                        MY_BUCKET,     /* The bucket to upload to */
//                        OBJECT_KEY + counter,    /* The key for the uploaded object */
//                        file        /* The file where the data to upload exists */
//                );
//                Log.d(TAG, "onPictureTaken: " + observer.getBucket());
//                Log.d(TAG, "onPictureTaken: " + observer.getKey());
                counter++;
                imageFileOS.flush();
                imageFileOS.close();
            } catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
            mCamera.startPreview();
        }
    };


    class CameraView extends SurfaceView implements SurfaceHolder.Callback {

        private SurfaceHolder mHolder;
        private Camera mCamera;
        private List<Camera.Size> mSupportedPreviewSizes;
        private Camera.Size mPreviewSize;

        public CameraView(Context context, Camera camera) {
            super(context);
            mCamera = camera;
            mCamera.setDisplayOrientation(90);

            mSupportedPreviewSizes = mCamera.getParameters().getSupportedPreviewSizes();

            //get the holder and set this class as the callback, so we can get camera data here
            mHolder = getHolder();
            mHolder.addCallback(this);
            mHolder.setType(SurfaceHolder.SURFACE_TYPE_NORMAL);
        }

        @Override
        public void surfaceCreated(SurfaceHolder surfaceHolder) {
            try {
                //when the surface is created, we can set the camera to draw images in this surfaceholder
                mCamera.setPreviewDisplay(surfaceHolder);
                mCamera.startPreview();
            } catch (IOException e) {
                Log.d("ERROR", "Camera error on surfaceCreated " + e.getMessage());
            }
        }

        @Override
        public void surfaceChanged(SurfaceHolder surfaceHolder, int i, int i2, int i3) {
            //before changing the application orientation, you need to stop the preview, rotate and then start it again
            if (mHolder.getSurface() == null)//check if the surface is ready to receive camera data
                return;

            try {
                mCamera.stopPreview();
            } catch (Exception e) {
                //this will happen when you are trying the camera if it's not running
            }

            //now, recreate the camera preview
            try {
                Camera.Parameters parameters = mCamera.getParameters();
                parameters.setPreviewSize(mPreviewSize.width, mPreviewSize.height);
                mCamera.setParameters(parameters);
                mCamera.setDisplayOrientation(90);
                mCamera.setPreviewDisplay(mHolder);
                mCamera.startPreview();
            } catch (IOException e) {
                Log.d("ERROR", "Camera error on surfaceChanged " + e.getMessage());
            }
        }

        @Override
        public void surfaceDestroyed(SurfaceHolder surfaceHolder) {
            //our app has only one screen, so we'll destroy the camera in the surface
            //if you are unsing with more screens, please move this code your activity
            mCamera.stopPreview();
            mCamera.release();
        }

        @Override
        protected void onMeasure(int widthMeasureSpec, int heightMeasureSpec) {
            final int width = resolveSize(getSuggestedMinimumWidth(), widthMeasureSpec);
            final int height = resolveSize(getSuggestedMinimumHeight(), heightMeasureSpec);

            if (mSupportedPreviewSizes != null) {
                mPreviewSize = getOptimalPreviewSize(mSupportedPreviewSizes, width, height);
            }

            if (mPreviewSize != null) {
                float ratio;
                if (mPreviewSize.height >= mPreviewSize.width)
                    ratio = (float) mPreviewSize.height / (float) mPreviewSize.width;
                else
                    ratio = (float) mPreviewSize.width / (float) mPreviewSize.height;

                // One of these methods should be used, second method squishes preview slightly
//                setMeasuredDimension(width, (int) (width * ratio));
                setMeasuredDimension((int) (width * ratio), height);
            }
        }

        private Camera.Size getOptimalPreviewSize(List<Camera.Size> sizes, int w, int h) {
            final double ASPECT_TOLERANCE = 0.1;
            double targetRatio = (double) h / w;

            if (sizes == null)
                return null;

            Camera.Size optimalSize = null;
            double minDiff = Double.MAX_VALUE;

            int targetHeight = h;

            for (Camera.Size size : sizes) {
                double ratio = (double) size.height / size.width;
                if (Math.abs(ratio - targetRatio) > ASPECT_TOLERANCE)
                    continue;

                if (Math.abs(size.height - targetHeight) < minDiff) {
                    optimalSize = size;
                    minDiff = Math.abs(size.height - targetHeight);
                }
            }

            if (optimalSize == null) {
                minDiff = Double.MAX_VALUE;
                for (Camera.Size size : sizes) {
                    if (Math.abs(size.height - targetHeight) < minDiff) {
                        optimalSize = size;
                        minDiff = Math.abs(size.height - targetHeight);
                    }
                }
            }

            return optimalSize;
        }
    }
}