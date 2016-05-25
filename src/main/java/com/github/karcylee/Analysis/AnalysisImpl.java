package com.github.karcylee.Analysis;

import org.ansj.app.keyword.KeyWordComputer;
import org.ansj.app.keyword.Keyword;
import org.ansj.domain.Term;
import org.ansj.splitWord.analysis.ToAnalysis;
import org.ansj.util.FilterModifWord;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import uk.org.lidalia.sysoutslf4j.context.LogLevel;
import uk.org.lidalia.sysoutslf4j.context.SysOutOverSLF4J;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by KarlLee on 2016/5/24.
 */
public class AnalysisImpl {

    private static Logger logger = LoggerFactory.getLogger(AnalysisImpl.class);

    public static List<String> doc2words(String doc){
        try{
            String  Content = doc2String(doc);
            List<String> result = str2words(Content);
            return result;
        }catch (Exception e){
            logger.error( "doc2words() error " ,e);
            return null;
        }
    }
    public static  List<String> str2words(String str){
        try {
            //分词器过滤词性
            FilterModifWord.insertStopNatures("v") ;
            FilterModifWord.insertStopNatures("w") ;
            FilterModifWord.insertStopNatures("uj") ;
            FilterModifWord.insertStopNatures("ul") ;
            FilterModifWord.insertStopNatures("m") ;
            FilterModifWord.insertStopNatures("en") ;
            FilterModifWord.insertStopNatures("null") ;
            FilterModifWord.insertStopNatures("p") ;
            FilterModifWord.insertStopWord("");
            //分词
            List<Term> TokensList = ToAnalysis.parse(str);
            //对分词结果进行过滤
            TokensList = FilterModifWord.modifResult(TokensList);
            //System.out.print(TokensList);
            List<String> result = new ArrayList<String>() ;
            for(Term t :TokensList){
                result.add(t.getName());
            }
            return result;
        }catch (Exception e){
            logger.error("str2words() error ",e);
            return null;
        }
    }
    public static List<Keyword> extractKeywords(String title, String content, int KeywordsNum) {
        try {
            KeyWordComputer kwc = new KeyWordComputer(KeywordsNum);
            List<Keyword> result = kwc.computeArticleTfidf(title, content);
        /*
        // 每个元素有 name,score,freq等成员。
        Iterator<Keyword> it = result.iterator();
        while (it.hasNext()) {
            Keyword tmp = it.next();
            System.out.printf("name: %s , score: %f , freq: %d  \n",tmp.getName(),tmp.getScore(),tmp.getFreq());
        }
        */
            return result;
        } catch (Exception e) {
            logger.error("extractKeywords() error ", e);
            return null;
        }
    }
    public static String doc2String(String docPath) {
        StringBuilder content = new StringBuilder();
        try {
            //String code = resolveCode(docPath); //计算的编码

            InputStream is = new FileInputStream(docPath);
            InputStreamReader isr = new InputStreamReader(is, "UTF-8");
            BufferedReader br = new BufferedReader(isr);

            String str = null;
            while (null != (str = br.readLine())) {
                content.append(str);
            }
            br.close();
        } catch (Exception e) {
            logger.error("Doc2String() error ！" +"读取文件:" + docPath + "失败!",e);
        }
        return content.toString();
    }

    /**
     * 判断并返回文本编码方式
     * @param path ： 文件路径
     * @return ： string，文件编码类型。
     */
    private static String resolveCode(String path) throws Exception {
//      String filePath = "D:/article.txt"; //[-76, -85, -71]  ANSI
//      String filePath = "D:/article111.txt";  //[-2, -1, 79] unicode big endian
//      String filePath = "D:/article222.txt";  //[-1, -2, 32]  unicode
//      String filePath = "D:/article333.txt";  //[-17, -69, -65] UTF-8
        try {
            InputStream inputStream = new FileInputStream(path);
            byte[] head = new byte[3];
            inputStream.read(head);
            String code = "gb2312";  //或GBK
            if (head[0] == -1 && head[1] == -2)
                code = "UTF-16";
            else if (head[0] == -2 && head[1] == -1)
                code = "Unicode";
            else if (head[0] == -17 && head[1] == -69 && head[2] == -65)
                code = "UTF-8";
            inputStream.close();
            //System.out.println(code);
            return code;
        }catch(Exception e){
            logger.error("resolveCode() error ",e);
            return "utf-8"; //默认utf-8
        }
    }

    public static void main(String[] args) {

        SysOutOverSLF4J.sendSystemOutAndErrToSLF4J(LogLevel.INFO, LogLevel.ERROR);

        String content = "           中国水法研究会在京成立\n" +
                "新华社北京５月１５日电（记者王坚）以增进对水的\n" +
                "             立法、政策和行政管理研究、促进水资源法制建设为宗旨\n" +
                "             的中国水法研究会今天在北京成立。全国政协副主席钱正\n" +
                "             英担任研究会的名誉会长。\n" +
                "                 钱正英在成立会上指出，贯彻《水法》，必须建立三\n" +
                "             个体系，即水法规体系、水管理体系和水执法体系。在这\n" +
                "             方面，有许多理论与实践的课题需要研究和探讨。中国水\n" +
                "             法研究会的成立，对促进我国水利法制建设工作具有重要\n" +
                "             意义。（完）\n" +
                "\n";
        List<Term> TokensList = ToAnalysis.parse(content);
        //对分词结果进行过滤
        TokensList = FilterModifWord.modifResult(TokensList);
        for (Term t : TokensList){
            System.out.println(t.getName()+ " " + t.termNatures() + " "+t.score());
        }
    }


}
