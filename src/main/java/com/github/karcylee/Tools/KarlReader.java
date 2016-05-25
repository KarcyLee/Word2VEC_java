package com.github.karcylee.Tools;

import java.io.*;

/**
 * Created by KarlLee on 2016/5/25.
 */
public class KarlReader {
    public static void main(String[] args) {
        KarlReader testReader = new KarlReader();
        String path = "./out/test";
        //testReader.readFileByFileReader(path);
        testReader.readFileByBufferedReader(path);

    }

    public void readFileByFileReader(String path){
        FileReader fileReader = null;
        try {
            fileReader = new FileReader(path);
            char[] buf = new char[1024]; //每次读取1024个字符
            int temp = 0;
            System.out.println("readFileByFileReader执行结果：");
            while ((temp = fileReader.read(buf)) != -1) {
                System.out.print(new String(buf, 0, temp));
            }
            System.out.println();
        } catch (Exception e) {
            e.printStackTrace();
        } finally { //像这种i/o操作尽量finally确保关闭
            if (fileReader!=null) {
                try {
                    fileReader.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    public void readFileByBufferedReader(String path){
        File file = new File(path);
        if (file.isFile()) {
            BufferedReader bufferedReader = null;
            FileReader fileReader = null;
            InputStreamReader isr  = null;
            try {
                fileReader = new FileReader(file);
                String encode = fileReader.getEncoding();
                isr = new InputStreamReader(new FileInputStream(file), encode);
                //bufferedReader = new BufferedReader(fileReader);
                bufferedReader = new BufferedReader(isr);
                String line = bufferedReader.readLine();
                System.out.println("readFileByBufferReader执行结果：");
                while (line != null) {
                    System.out.println(line);
                    line = bufferedReader.readLine();
                }
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            } catch (IOException e) {
                e.printStackTrace();
            } finally {
                try {
                    fileReader.close();
                    bufferedReader.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

}
