# 可以自己import我们平台支持的第三方python模块，比如pandas、numpy等。
# 回测区间：2014-01-01~2018-01-01
# 选股：
# 范围：沪深300
# 因子：pe_ratio,pb_ratio,market_cap,ev,return_on_asset_net_profit,du_return_on_equity,earnings_per_share,revenue,total_expense,
# 方法：回归法，系数相乘（矩阵相乘运算）得出结果排序
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


# 在这个方法中编写任何的初始化逻辑。context对象将会在你的算法策略的任何方法之间做传递。
def init(context):
    # 定义购买的股票数量
    context.stock_num = 20

    # 定义沪深300指数股
    context.hs300 = index_components("000300.XSHG")

    # 回归系数
    context.weights = np.array([-0.01864979, -0.04537212, -0.18487143, -0.06092573,  0.18599453,-0.02088234,  0.03341527,  0.91743347, -0.8066782 ])


    # 回归选股法 定时函数（每月）
    scheduler.run_monthly(regression_select, tradingday=1)


def regression_select(context, bar_dict):
    """回归法进行选择股票
    准备因子数据、数据处理(缺失值、去极值、标准化、中性化)
    预测每个股票对应这一天的结果，然后排序选出前20只股票
    """
    # 1、准备因子数据
    q  = query(fundamentals.eod_derivative_indicator.pe_ratio,
           fundamentals.eod_derivative_indicator.pb_ratio,
           fundamentals.eod_derivative_indicator.market_cap,
           fundamentals.financial_indicator.ev,
           fundamentals.financial_indicator.return_on_asset_net_profit,
           fundamentals.financial_indicator.du_return_on_equity,
           fundamentals.financial_indicator.earnings_per_share,
           fundamentals.income_statement.revenue,
           fundamentals.income_statement.total_expense).filter(fundamentals.stockcode.in_(context.hs300))

    fund = get_fundamentals(q)

    context.factors_data = fund.T

    # 2、处理数据函数
    dealwith_data(context)

    # 3、选股函数(每月调整股票池)
    select_stocklist(context)

    # 4、定期调仓函数
    rebalance(context)


def dealwith_data(context):
    """
    处理因子数据函数
    准备因子数据、数据处理(缺失值、去极值、标准化、中性化)
    """
    # 缺失值
    context.factors_data = context.factors_data.dropna()
    # 保留原来的数据，后续处理
    x = context.factors_data["market_cap"]

    # 循环处理因子数据
    for name in context.factors_data.columns:

        # 去极值、标准化
        context.factors_data[name] = mad(context.factors_data[name])
        context.factors_data[name] = stand(context.factors_data[name])

        # 市值中性化处理
        # x 市值原始数据
        # y 对应因子的处理好的数据、经过了去极值和标准化后的因子值
        if name == "market_cap":
            continue
        # 取出因子作为目标值
        y = context.factors_data[name]

        # 建立回归方程，得出预测结果
        # 用真是结果-预测结果得到残差，即为新的因子值
        lr = LinearRegression()

        lr.fit(x.values.reshape(-1, 1), y.values)

        # 预测结果
        y_predict = lr.predict(x.values.reshape(-1, 1))

        # 得出没有相关性的残差部分
        # 将残差部分作为新的因子值
        context.factors_data[name] = y - y_predict


def select_stocklist(context):
    """选择股票池函数
    """

    # 建立回归方程，得出预测结果，然后排序选出30个股票
    # 特征值：factors_data：300只股票9个因子的特征值
    # 训练的权重系数为：9个权重
    # 假如5月1日调完仓，
    # 得出的结果：相当于预测接下来的5月份收益率，哪个收益率高选谁

    # 进行特征值与权重之间的矩阵运算
    # (m行，n列) *(n行,l列) = (m行，l列)
    # (300, 9) * (9, 1) = (300, 1)

    mat_res = np.dot(context.factors_data.values, context.weights)

    context.factors_data['return'] = mat_res

    # 按照回归结果排序，预测收益高的作为股票池
    context.stock_list = context.factors_data.sort_values(by="return", ascending=False).index[:context.stock_num]


def rebalance(context):
    """定期调仓函数
    """
    # 卖出
    for stock in context.portfolio.positions.keys():

        if context.portfolio.positions[stock].quantity > 0:

            if stock not in context.stock_list:

                order_target_percent(stock, 0)

    weight = 1.0/len(context.stock_list)
    # 买入
    for stock in context.stock_list:

        order_target_percent(stock, weight)


# before_trading此函数会在每天策略交易开始前被调用，当天只会被调用一次
def before_trading(context):
    pass


# 你选择的证券的数据更新将会触发此段逻辑，例如日或分钟历史数据切片或者是实时数据切片更新
def handle_bar(context, bar_dict):
    pass

# after_trading函数会在每天交易结束后被调用，当天只会被调用一次
def after_trading(context):
    pass


def mad(factor):
    """3倍中位数去极值
    """
    # 求出因子值的中位数
    med = np.median(factor)

    # 求出因子值与中位数的差值，进行绝对值
    mad = np.median(np.abs(factor - med))

    # 定义几倍的中位数上下限
    high = med + (3 * 1.4826 * mad)
    low = med - (3 * 1.4826 * mad)

    # 替换上下限以外的值
    factor = np.where(factor > high, high, factor)
    factor = np.where(factor < low, low, factor)

    return factor


def stand(factor):
    """标准化
    """
    mean = np.mean(factor)
    std = np.std(factor)
    return (factor - mean)/std