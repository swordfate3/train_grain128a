# 设置输出为PNG格式
set terminal png size 800,600
set output 'result.png'

# 设置图表标题和轴标签
set title ""
set xlabel "rounds Axis"
set ylabel "accuracy Axis"

# 设置网格线
set grid
set style fill transparent solid 0.6 border
set yrange [0:1.2]
# 定义第一条曲线，不填充
plot "result" using 1:2 with lp title 'accuracy'

# 重置终端设置，以便在命令行界面中继续绘图
set terminal x11
set output
