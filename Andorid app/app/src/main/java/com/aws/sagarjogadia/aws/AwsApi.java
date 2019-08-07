package com.aws.sagarjogadia.aws;

import retrofit2.Call;
import retrofit2.http.Body;
import retrofit2.http.POST;

/**
 * Created by Sagar Jogadia on 06/05/2017.
 */

interface AwsApi {

    //http://34.206.58.98:8000/train
    @POST("test")
    Call<AwsResponse> sendData(@Body AwsModel awsModel);
}
