package com.aws.sagarjogadia.aws;

import com.google.gson.annotations.Expose;
import com.google.gson.annotations.SerializedName;

import javax.annotation.Generated;

/**
 * Created by Sagar Jogadia on 06/05/2017.
 */

@Generated("org.jsonschema2pojo")
public class AwsModel {

    @SerializedName("direction")
    @Expose
    private String direction;

    @SerializedName("image_data")
    @Expose
    private String imageData;

    @SerializedName("filename")
    @Expose
    private String fileName;

    public String getDirection() {
        return direction;
    }

    public void setDirection(String direction) {
        this.direction = direction;
    }

    public String getImageData() {
        return imageData;
    }

    public void setImageData(String imageData) {
        this.imageData = imageData;
    }

    public String getFileName() {
        return fileName;
    }

    public void setFileName(String fileName) {
        this.fileName = fileName;
    }
}
