package com.github.karcylee.Tools;

import java.io.*;
import java.util.Vector;


public class KarlSerializationHelper  {

    public KarlSerializationHelper() {
    }

    public static void write(String filename, Object o) throws Exception {
        write((OutputStream)(new FileOutputStream(filename)), o);
    }

    public static void write(OutputStream stream, Object o) throws Exception {
        if(!(stream instanceof BufferedOutputStream)) {
            stream = new BufferedOutputStream((OutputStream)stream);
        }

        ObjectOutputStream oos = new ObjectOutputStream((OutputStream)stream);
        oos.writeObject(o);
        oos.flush();
        oos.close();
    }

    public static void writeAll(String filename, Object[] o) throws Exception {
        writeAll((OutputStream)(new FileOutputStream(filename)), o);
    }

    public static void writeAll(OutputStream stream, Object[] o) throws Exception {
        if(!(stream instanceof BufferedOutputStream)) {
            stream = new BufferedOutputStream((OutputStream)stream);
        }

        ObjectOutputStream oos = new ObjectOutputStream((OutputStream)stream);

        for(int i = 0; i < o.length; ++i) {
            oos.writeObject(o[i]);
        }

        oos.flush();
        oos.close();
    }

    public static Object read(String filename) throws Exception {
        return read((InputStream)(new FileInputStream(filename)));
    }

    public static Object read(InputStream stream) throws Exception {
        if(!(stream instanceof BufferedInputStream)) {
            stream = new BufferedInputStream((InputStream)stream);
        }

        ObjectInputStream ois = new ObjectInputStream((InputStream)stream);
        Object result = ois.readObject();
        ois.close();
        return result;
    }

    public static Object[] readAll(String filename) throws Exception {
        return readAll((InputStream)(new FileInputStream(filename)));
    }

    public static Object[] readAll(InputStream stream) throws Exception {
        if(!(stream instanceof BufferedInputStream)) {
            stream = new BufferedInputStream((InputStream)stream);
        }

        ObjectInputStream ois = new ObjectInputStream((InputStream)stream);
        Vector result = new Vector();

        try {
            while(true) {
                result.add(ois.readObject());
            }
        } catch (IOException var4) {
            ois.close();
            return result.toArray(new Object[result.size()]);
        }
    }



    public static void main(String[] args) throws Exception {

        float [] data = {1,2,3,4,5,6};

        String name = "xuliehua";
        KarlSerializationHelper.write(name,data);
        float[] res = (float [])KarlSerializationHelper.read(name);
        for (float a : res){
            System.out.println(a);
        }



    }
}
