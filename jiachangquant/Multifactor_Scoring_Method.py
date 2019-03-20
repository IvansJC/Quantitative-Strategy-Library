# 可以自己import我们平台支持的第三方python模块，比如pandas、numpy等。
# 打分法选股
# 回测：2010-01-01~2018-01-01
# 调仓：按月
# 选股因子：市值-market_cap、市盈率-pe_ratio、市净率-pb_ratio、ROIC-return_on_invested_capital、inc_revenue-营业总收入   和inc_profit_before_tax-利润增长率
# 选股的指数、模块：全A股
import pandas as pd


# 在这个方法中编写任何的初始化逻辑。context对象将会在你的算法策略的任何方法之间做传递。
def init(context):
    context.group_number = 10
    context.stocknum = 20

    # 定义调仓频率函数
    scheduler.run_monthly(score_select, tradingday=1)


def score_select(context, bar_dict):
    """打分法选股逻辑
    """

    # 选股逻辑，获取数据、数据处理、选股方法

    # 1、获取选股的因子数据
    q = query(fundamentals.eod_derivative_indicator.market_cap,
              fundamentals.eod_derivative_indicator.pe_ratio,
              fundamentals.eod_derivative_indicator.pb_ratio,
              fundamentals.financial_indicator.return_on_invested_capital,
              fundamentals.financial_indicator.inc_revenue,
              fundamentals.financial_indicator.inc_profit_before_tax
              )

    fund = get_fundamentals(q)

    # 通过转置将股票索引变成行索引、指标变成列索引
    factors_data = fund.T

    # 数据处理
    factors_data = factors_data.dropna()

    # 对每个因子进行打分估计，得出综合评分
    get_score(context, factors_data)

    # # 进行调仓逻辑，买卖
    rebalance(context)


def get_score(context, factors_data):
    """
    对因子选股数据打分
    因子升序：市值、市盈率、市净率
    因子降序：ROIC、inc_revenue营业总收入和inc_profit_before_tax利润增长率
    """
    # logger.info(factors_data)
    for factorname in factors_data.columns:

        if factorname in ['pe_ratio', 'pb_ratio', 'market_cap']:
            # 单独取出每一个因子去进行处理分组打分
            factor = factors_data.sort_values(by=factorname)[factorname]
        else:

            factor = factors_data.sort_values(by=factorname, ascending=False)[factorname]
            # logger.info(factor)

        # 对于factor转换成dataframe，为了新增列分数
        factor = pd.DataFrame(factor)

        factor[factorname + "score"] = 0

        # logger.info(factor)
        # 对于每个因子去进行分组打分
        # 求出所有的股票个数
        single_groupnum = len(factor) // 10

        for i in range(10):

            if i == 9:
                factor[factorname + "score"][i * single_groupnum:] = i + 1

            factor[factorname + "score"][i * single_groupnum:(i + 1) * single_groupnum] = i + 1

        # 打印分数
        # logger.info(factor)
        # 拼接每个因子的分数到原始的factors_data数据当中
        factors_data = pd.concat([factors_data, factor[factorname + "score"]], axis=1)

    # logger.info(factors_data)

    # 求出总分
    # 先取出这几列分数
    # 求和分数
    sum_score = factors_data[
        ["market_capscore", "pe_ratioscore", "pb_ratioscore", "return_on_invested_capitalscore", "inc_revenuescore",
         "inc_profit_before_taxscore"]].sum(1).sort_values()

    # 拼接到factors_data
    # logger.info(sum_score)
    context.stocklist = sum_score[:context.stocknum].index


def rebalance(context):
    # 卖出
    for stock in context.portfolio.positions.keys():

        if context.portfolio.positions[stock].quantity > 0:

            if stock not in context.stocklist:
                order_target_percent(stock, 0)

    # 买入
    for stock in context.stocklist:
        order_target_percent(stock, 1.0 / 20)


# before_trading此函数会在每天策略交易开始前被调用，当天只会被调用一次
def before_trading(context):
    pass


# 你选择的证券的数据更新将会触发此段逻辑，例如日或分钟历史数据切片或者是实时数据切片更新
def handle_bar(context, bar_dict):
    # 开始编写你的主要的算法逻辑
    pass


# after_trading函数会在每天交易结束后被调用，当天只会被调用一次
def after_trading(context):
    pass