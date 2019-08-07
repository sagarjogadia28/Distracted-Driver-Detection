package com.aws.sagarjogadia.aws;

import android.Manifest;
import android.annotation.TargetApi;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.widget.Button;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "permission";
    private static final String CAMERA = Manifest.permission.CAMERA;
    private static final String WRITE_STORAGE = Manifest.permission.WRITE_EXTERNAL_STORAGE;
    private static final int REQUEST_MULTIPLE = 1;
//    int x, y;
//    int pos;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Button mCameraButton = (Button) findViewById(R.id.buttonCamera);

//        Spinner spinner = (Spinner) findViewById(R.id.spinner);
//        ArrayAdapter<CharSequence> adapter = ArrayAdapter.createFromResource(this,
//                R.array.camera_options, android.R.layout.simple_spinner_item);
//        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
//        spinner.setAdapter(adapter);
//
//        EditText editTextX = (EditText) findViewById(R.id.editX);
//        EditText editTextY = (EditText) findViewById(R.id.editY);
//
//        x = Integer.valueOf(editTextX.getText().toString());
//        y = Integer.valueOf(editTextY.getText().toString());

        checkMultiplePermissions();


        mCameraButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(getApplicationContext(), CameraActivity.class);
//                intent.putExtra("Xvalue", x);
//                intent.putExtra("Yvalue", y);

                startActivity(intent);

            }
        });
    }

    private void checkMultiplePermissions() {
        int permissionCamera = ContextCompat.checkSelfPermission(this,
                CAMERA);
        int writeStorage = ContextCompat.checkSelfPermission(this, WRITE_STORAGE);

        List<String> multiplePermissions = new ArrayList<>();
        if (permissionCamera != PackageManager.PERMISSION_GRANTED)
            multiplePermissions.add(CAMERA);
        if (writeStorage != PackageManager.PERMISSION_GRANTED)
            multiplePermissions.add(WRITE_STORAGE);

        if (!multiplePermissions.isEmpty())
            ActivityCompat.requestPermissions(this, multiplePermissions.toArray(new String[multiplePermissions.size()]), REQUEST_MULTIPLE);
    }


    @TargetApi(Build.VERSION_CODES.M)
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {

        if (requestCode == REQUEST_MULTIPLE) {

            Map<String, Integer> permissionMap = new HashMap<>();
            permissionMap.put(CAMERA, PackageManager.PERMISSION_GRANTED);
            permissionMap.put(WRITE_STORAGE, PackageManager.PERMISSION_GRANTED);

            if (grantResults.length > 0) {
                for (int i = 0; i < permissions.length; i++) {
                    Log.d(TAG, "onRequestPermissionsResult: PERMISSION: " + permissions[i] + " GRANT:" + grantResults[i]);
                    permissionMap.put(permissions[i], grantResults[i]);
                }

                if (permissionMap.get(CAMERA) == PackageManager.PERMISSION_GRANTED
                        && permissionMap.get(WRITE_STORAGE) == PackageManager.PERMISSION_GRANTED) {
                    Log.d(TAG, "Granted");
                }
//                else {
//                    if (ActivityCompat.shouldShowRequestPermissionRationale(this, CAMERA) || ActivityCompat.shouldShowRequestPermissionRationale(this, WRITE_STORAGE)  || ActivityCompat.shouldShowRequestPermissionRationale(this, READ_STORAGE)) {
//                        //createSnackbar
//
//                    } else {
//                        //goto settings
//
//                    }
//                }
            }
        }

    }

//    @Override
//    public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
//         pos = position;
//    }
//
//    @Override
//    public void onNothingSelected(AdapterView<?> parent) {
//        pos = 1;
//    }
}

