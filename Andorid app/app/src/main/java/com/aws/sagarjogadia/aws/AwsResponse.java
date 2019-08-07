package com.aws.sagarjogadia.aws;

import com.google.gson.annotations.Expose;
import com.google.gson.annotations.SerializedName;

import javax.annotation.Generated;

/**
 * Created by Sagar Jogadia on 07/05/2017.
 */
@Generated("org.jsonschema2pojo")
public class AwsResponse {

    @SerializedName("message")
    @Expose
    private String message;

    @SerializedName("filename")
    @Expose
    private String fileName;

    @SerializedName("distraction_type")
    @Expose
    private int distractedType;

    public String getFileName() {
        return fileName;
    }

    public void setFileName(String fileName) {
        this.fileName = fileName;
    }

    public int getDistractedType() {
        return distractedType;
    }

    public void setDistractedType(int distractedType) {
        this.distractedType = distractedType;
    }

    public String getMessage() {
        return message;
    }

    public void setMessage(String message) {
        this.message = message;
    }

}
