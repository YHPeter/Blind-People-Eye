package com.maishack.network;

import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Environment;

import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.net.HttpURLConnection;
import java.net.URI;
import java.net.URL;

public class UploadFile {
    public static String upload(File file) throws Exception{
        HttpURLConnection conn = (HttpURLConnection) new URL("http://159.75.113.35:8123/model").openConnection();
        String boundary_string = "asdlflh8wf8lwhf03wf39whf";

        // we want to write out
        conn.setDoOutput(true);
        conn.setRequestMethod("POST");
        conn.addRequestProperty("Content-Type", "multipart/form-data; boundary=" + boundary_string);

        // now we write out the multipart to the body
        OutputStream conn_out = conn.getOutputStream();
        BufferedWriter conn_out_writer = new BufferedWriter(new OutputStreamWriter(conn_out));
        conn_out_writer.write("\r\n--" + boundary_string + "\r\n"); // "\r\n--" + boundary_string +
        conn_out_writer.write("Content-Disposition: form-data; " + "name=\"image_file\"; " + "filename=\""
                + file.getName() + "\"" + "\r\n\r\n");
        conn_out_writer.flush();

        // payload from the file
        FileInputStream file_stream = new FileInputStream(file);
        // write direct to outputstream instance, because we write now bytes and not
        // strings
        int read_bytes;
        byte[] buffer = new byte[1024 * 1024];
        while ((read_bytes = file_stream.read(buffer)) != -1) {
            conn_out.write(buffer, 0, read_bytes);
        }
        conn_out.flush();
        // close multipart body
        conn_out_writer.write("\r\n--" + boundary_string + "--\r\n");
        conn_out_writer.flush();

        // close all the streams
        conn_out_writer.close();
        conn_out.close();
        file_stream.close();
        // execute and get response code
        conn.getResponseCode();

        try (BufferedReader br = new BufferedReader(new InputStreamReader(conn.getInputStream(), "utf-8"))) {
            StringBuilder response = new StringBuilder();
            String responseLine = null;
            while ((responseLine = br.readLine()) != null) {
                response.append(responseLine.trim());
            }
            String res = response.toString();
            return res.substring(10, res.length() - 2);
        }
    }
}
