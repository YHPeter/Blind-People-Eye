import java.io.BufferedWriter;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.net.HttpURLConnection;
import java.net.URL;

//import org.json.JSONObject;
class Client {
    private static String upload(String image_adress) throws Exception {
        File file = new File(image_adress);
        HttpURLConnection conn = (HttpURLConnection) new URL("http://159.75.113.35:8123/model").openConnection();
        String boundary_string = "asdlflh8wf8lwhf03wf39whf";

        conn.setDoOutput(true);
        conn.setRequestMethod("POST");
        conn.addRequestProperty("Content-Type", "multipart/form-data; boundary=" + boundary_string);

        OutputStream conn_out = conn.getOutputStream();
        BufferedWriter conn_out_writer = new BufferedWriter(new OutputStreamWriter(conn_out));
        conn_out_writer.write("\r\n--" + boundary_string + "\r\n");
        conn_out_writer.write("Content-Disposition: form-data; " + "name=\"image_file\"; " + "filename=\""
                + file.getName() + "\"" + "\r\n\r\n");
        conn_out_writer.flush();

        FileInputStream file_stream = new FileInputStream(file);
        int read_bytes;
        byte[] buffer = new byte[1024 * 1024];
        while ((read_bytes = file_stream.read(buffer)) != -1) {
            conn_out.write(buffer, 0, read_bytes);
        }
        conn_out.flush();
        conn_out_writer.write("\r\n--" + boundary_string + "--\r\n");
        conn_out_writer.flush();

        conn_out_writer.close();
        conn_out.close();
        file_stream.close();
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

    public static void main(String[] args) throws Exception {
        System.out.println(upload("test.jpg"));
    }
}
