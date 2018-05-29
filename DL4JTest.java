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

		String[] test = {"周静  16 年 在 营收 快速 增长 的 情况 下 三 项 费用 得到 了 较 好 的 控制  其中 利息 支出 为 6366万  较 去年 同 期 较少 近 2000万元  显示 出 公司 的 财务 状况 持续 好转  另外 有 汇兑 收益 1890万元  与 去年 基本 持平  管理 费用  销售 费用 分别 增长 14.29%  10.75%  低于 营收 增速  显示 出 公司 在 内部 治理 水平 上 的 不断 提升  其中 研发 费用 1.1亿  较 去年 同 期 增加 0.28亿  营收 占 比 达到 2.74%  显示 出 公司 对 研发 的 重视",
				"文献 伊利 系统性 作战 能力 行业 领先  各 业务 线 市场 份额 持续 提升  品牌 & 渠道 优势 明显  费用率 有 压缩 空间  净利 持续 增长 可 期  17年 不 到 20 倍 PE  仍 值得 拥有  维持  强烈 推荐  评级",
				"彭琦 对于 FC 封装 而言  Bumping 技术 是 其 前端 核心 封装 工艺 和 技术 门槛 点  封装 厂 拥有 自主 的 Bumping 量产 能力  对于 其 获取 FC  WLP 等 先进 封装 订单 具有 重要 意义  昆山 华天 在 Bumping 等 前道 工艺 量产 能力 的 快速 提升  对于 西安 FC 等 后 道 封装 业务量 提升 将 形成 有力 支撑 ",
				"石亮  2016 第四 季度 计提 地 博 矿业 的 投资 跌价 准备 影响 了 业绩  2017 恢复 正常 公司 2016 第四 季度 的 净利润 仅 2326 万元  主要 是 2016第四季度公司 提取 了 地 博 矿业 的 投资 跌价 准备 4576万元  至此 公司 对地 博 矿业 的 投资 已经 全部 计提 了 跌价 准备  未来 公司 业绩 将 不再 受 此 影响  抛开 资产 减值 损失 等 因素 的 影响  2016 第四 季度 公司 毛利 达到 1.5亿元  环比 第三 季度 略有 下滑  但 同比 增长 23%  可以 说 公司 2016 第四 季度 处于 良好 的 增长 通道",
				"石亮 涤纶 工业 丝 销量 增长 及 毛利率 提升 至 业绩 大幅 增长 2016 年 公司 新 投产 1.5万 吨 高模低 收缩 丝  2万 吨 安全带 丝 和 2万 吨 安全 气囊 丝  导致 公司 车用 丝 占 比 从 2015 的 55% 提升 至 2016年 的 67%  5.5万 吨 车用 丝 的 投产 导致 公司 2016 年 涤纶 工业 丝 产销量 的 增长  公司 营业 收入 同比 增长 21.01%  在 销量 增长 的 同时  公司 车用 丝 占 比 的 提升 使得 公司 综合 毛利率 有所 提升  2016 年 公司 毛利率 达到 23.7%  同比 提升 1.6%  分 产品 看  公司 帘子布 毛利率 大幅 提升 10.18% 至 19.21%  涤纶 工业 丝毛 利率 提升 2.3% 至 27.81%  在 竞争 加剧 的 情况 下  公司 灯箱 布和 PVC 膜 的 毛利率 均 有所 下滑  其中 灯箱 布 毛利率 下降 1.71% 至 18.22%  PVC 膜 毛利率 下滑 4.86% 至 19.41%"
				};
		for (String t:test){
			System.out.println(t);
			INDArray vector = ((ParagraphVectors)wv).inferVector(t);
			System.out.println(vector.toString());
		}
	}
}
