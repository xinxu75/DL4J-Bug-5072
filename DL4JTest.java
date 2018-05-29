import org.apache.commons.lang3.StringUtils;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.nio.charset.Charset;
import java.util.*;

public class DL4JTest {
	public static void main(String[] args) throws Exception {
      		WordVectors wv = WordVectorSerializer.readParagraphVectors("V3_PVec_ver1.tar.gz");
		DefaultTokenizerFactory tf = new DefaultTokenizerFactory();
	    	tf.setTokenPreProcessor(new CommonPreprocessor());
	    	((ParagraphVectors)wv).setTokenizerFactory(tf);

		String[] test = {"�ܾ�  16 �� �� Ӫ�� ���� ���� �� ��� �� �� �� ���� �õ� �� �� �� �� ����  ���� ��Ϣ ֧�� Ϊ 6366��  �� ȥ�� ͬ �� ���� �� 2000��Ԫ  ��ʾ �� ��˾ �� ���� ״�� ���� ��ת  ���� �� ��� ���� 1890��Ԫ  �� ȥ�� ���� ��ƽ  ���� ����  ���� ���� �ֱ� ���� 14.29%  10.75%  ���� Ӫ�� ����  ��ʾ �� ��˾ �� �ڲ� ���� ˮƽ �� �� ���� ����  ���� �з� ���� 1.1��  �� ȥ�� ͬ �� ���� 0.28��  Ӫ�� ռ �� �ﵽ 2.74%  ��ʾ �� ��˾ �� �з� �� ����",
				"���� ���� ϵͳ�� ��ս ���� ��ҵ ����  �� ҵ�� �� �г� �ݶ� ���� ����  Ʒ�� & ���� ���� ����  ������ �� ѹ�� �ռ�  ���� ���� ���� �� ��  17�� �� �� 20 �� PE  �� ֵ�� ӵ��  ά��  ǿ�� �Ƽ�  ����",
				"���� ���� FC ��װ ����  Bumping ���� �� �� ǰ�� ���� ��װ ���� �� ���� �ż� ��  ��װ �� ӵ�� ���� �� Bumping ���� ����  ���� �� ��ȡ FC  WLP �� �Ƚ� ��װ ���� ���� ��Ҫ ����  ��ɽ ���� �� Bumping �� ǰ�� ���� ���� ���� �� ���� ����  ���� ���� FC �� �� �� ��װ ҵ���� ���� �� �γ� ���� ֧�� ",
				"ʯ��  2016 ���� ���� ���� �� �� ��ҵ �� Ͷ�� ���� ׼�� Ӱ�� �� ҵ��  2017 �ָ� ���� ��˾ 2016 ���� ���� �� ������ �� 2326 ��Ԫ  ��Ҫ �� 2016���ļ��ȹ�˾ ��ȡ �� �� �� ��ҵ �� Ͷ�� ���� ׼�� 4576��Ԫ  ���� ��˾ �Ե� �� ��ҵ �� Ͷ�� �Ѿ� ȫ�� ���� �� ���� ׼��  δ�� ��˾ ҵ�� �� ���� �� �� Ӱ��  �׿� �ʲ� ��ֵ ��ʧ �� ���� �� Ӱ��  2016 ���� ���� ��˾ ë�� �ﵽ 1.5��Ԫ  ���� ���� ���� ���� �»�  �� ͬ�� ���� 23%  ���� ˵ ��˾ 2016 ���� ���� ���� ���� �� ���� ͨ��",
				"ʯ�� ���� ��ҵ ˿ ���� ���� �� ë���� ���� �� ҵ�� ��� ���� 2016 �� ��˾ �� Ͷ�� 1.5�� �� ��ģ�� ���� ˿  2�� �� ��ȫ�� ˿ �� 2�� �� ��ȫ ���� ˿  ���� ��˾ ���� ˿ ռ �� �� 2015 �� 55% ���� �� 2016�� �� 67%  5.5�� �� ���� ˿ �� Ͷ�� ���� ��˾ 2016 �� ���� ��ҵ ˿ ������ �� ����  ��˾ Ӫҵ ���� ͬ�� ���� 21.01%  �� ���� ���� �� ͬʱ  ��˾ ���� ˿ ռ �� �� ���� ʹ�� ��˾ �ۺ� ë���� ���� ����  2016 �� ��˾ ë���� �ﵽ 23.7%  ͬ�� ���� 1.6%  �� ��Ʒ ��  ��˾ ���Ӳ� ë���� ��� ���� 10.18% �� 19.21%  ���� ��ҵ ˿ë ���� ���� 2.3% �� 27.81%  �� ���� �Ӿ� �� ��� ��  ��˾ ���� ���� PVC Ĥ �� ë���� �� ���� �»�  ���� ���� �� ë���� �½� 1.71% �� 18.22%  PVC Ĥ ë���� �»� 4.86% �� 19.41%"
				};
		for (String t:test){
			System.out.println(t);
			INDArray vector = ((ParagraphVectors)wv).inferVector(t);
			System.out.println(vector.toString());
		}
	}
}
