package com.github.karcylee.Analysis;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import uk.org.lidalia.sysoutslf4j.context.LogLevel;
import uk.org.lidalia.sysoutslf4j.context.SysOutOverSLF4J;

import java.io.EOFException;
import java.io.File;
import java.io.RandomAccessFile;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by KarlLee on 2016/5/24.
 */
public class PrepareData {
    private static Logger logger = LoggerFactory.getLogger(PrepareData.class);

    final static List<File> getAllFiles(File dir) {
        List<File> result = new ArrayList<File>();
        File[] fs = dir.listFiles();
        for(int i=0; i < fs.length; ++i){
            result.add(fs[i]);
            //System.out.println(fs[i].getAbsolutePath());
            if(fs[i].isDirectory()){
                try{
                    List<File> child_files =getAllFiles(fs[i]);
                    for (File f : child_files){
                        result.add(f);
                    }
                }catch(Exception e){
                    e.printStackTrace();
                }
            }
        }
        return result;
    }

    //将文件夹下的文件都进行分词 空格分隔 每行一个文档
    public static void docs2txt(String folder, String output){
        List<File> allDocs = getAllFiles(new File(folder));
        long seek = 0;
        try {
            for (int i = 0; i < allDocs.size(); ++i) {
                if (allDocs.get(i).isFile()) {
                    String absName = allDocs.get(i).getAbsolutePath();
                    logger.info(absName);
                    List<String> words = AnalysisImpl.doc2words(absName);
                    RandomAccessFile raf = new RandomAccessFile(output, "rw");
                    raf.seek(seek);
                    for (int j = 0; j < words.size(); ++j) {
                        //logger.info(words.get(j));
                        //raf.writeUTF(words.get(j));
                        //raf.writeUTF(" ");
                        String s = words.get(j) + " ";
                        byte [] bytes = s.getBytes();
                        raf.write(bytes);
                    }
                    //raf.writeUTF("\n");
                    raf.writeByte('\n');
                    seek = raf.getFilePointer();
                }
            }
        }catch (Exception e){
            logger.error("",e);
        }
    }

    public static String readWord(RandomAccessFile raf) throws EOFException {
        String word = null;
        byte []bytes = new byte[1000];
        int len = 0;
        try {
            while (true) {
                byte ch = raf.readByte();
                if (ch == 13) continue;
                if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
                    if (len > 0) {
                        if (ch == '\n') {
                            //回退一格
                            long pos = raf.getFilePointer();
                            raf.seek(pos - 1);
                        }
                        word = new String(bytes, 0, len, StandardCharsets.UTF_8);
                        break;
                    }
                    if (ch == '\n') {
                        word = "</s>";
                        break;
                    } else
                        continue;
                }
                bytes[len++] = ch;
                if (len >= 10000 - 1)
                    --len;   // Truncate too long words
            }
        }catch (EOFException EOF){
            throw new EOFException("EOF");
        }catch (Exception e){
            logger.error("",e);
        }
        return word;
    }
    public static void recur_readWord(String input){
        try {
            RandomAccessFile raf = new RandomAccessFile(input, "r");
            while (true) {
                String s = readWord(raf);
                logger.info(s);
            }
        }catch (EOFException eof){
            logger.info("EOF");
        } catch (Exception e){
            logger.error("",e);
        }
    }

    public static void main(String[] args) {
        SysOutOverSLF4J.sendSystemOutAndErrToSLF4J(LogLevel.INFO, LogLevel.ERROR);

        docs2txt("D:\\Data\\Corpus\\tc-corpus-answer\\answer","./out/docs");

        //recur_readWord("./out/docs");
    }
}
