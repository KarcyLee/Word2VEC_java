import java.util.ArrayList;
import java.util.List;

/**
 * Created by KarlLee on 2016/5/25.
 */
public class javaTest {
    public static void main(String []args){
        List<NE> arr = new ArrayList<NE>();
        for(int i = 0; i < 3; ++ i ){
            NE t = new NE(i + 1);
            arr.add(t);
        }

        for (int j = 0 ; j < arr.size(); ++j){
            NE t = arr.get(j);
            for(int i = 0; i < t.a.length; ++i){
                t.a[i] = 999;
            }
        }
        for (int j = 0 ; j < arr.size(); ++j){
            NE t = arr.get(j);
            System.out.println(t.toString());
        }


    }

    public static class NE{
        int [] a = new int[10];
        public void setA(int k){
            for (int i = 0; i < a.length; ++ i){
                a[i] = k;
            }
        }
        public NE(int k){
            setA(k);
        }


        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < a.length; ++i){
                sb.append(Integer.toString(a[i]) + " ");
            }
            return  sb.toString();
        }
    }

}
