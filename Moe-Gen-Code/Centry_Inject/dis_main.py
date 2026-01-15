import os
import sys
project_path='/home/xuke/zpc/Code/Moe_Gen'
sys.path.append(project_path)


from CenterInject_Explore.tool_code import *
from CenterInject_Explore.Distribution.dis_fun import *
# Add heatmap plotting
import matplotlib.pyplot as plt
import seaborn as sns
from Defence_Method.AWA.awa_def import Eva_awa_CW
from Defence_Method.Alert.Alert_def import Alert_def_cw
from Defence_Method.Moe_Gen.Moe_Gen_Def import Moe_Gen_CW_Def
# Preprocessing setup
DataSet_Name='AWF100'  # Dataset name can be AWF100, AWF200, AWF500, AWF900  DF is not yet experimented
# Load original data
from DataTool_Code.LoadData import Load_AWF100_cw_test100
WF_modelName='AWF'  # Model name can be DF, AWF, VarCNN

data_x,label_y=Load_AWF100_cw_test100()

awa_data=Eva_awa_CW(data_x,label_y,WF_modelName)


alert_data=Alert_def_cw(data_x,label_y,WF_modelName)
moe_data=Moe_Gen_CW_Def(data_x,label_y,WF_modelName,DataSet_Name)

run_full_evaluation(data_x,awa_data,label_y,'awa')
run_full_evaluation(data_x,alert_data,label_y,'alert')
run_full_evaluation(data_x,moe_data,label_y,'moe-gen')

# Plot heatmap
def plot_ecdf_heatmap(ecdf_result, method_name="Defence Method"):
    # 1. Convert dictionary to 2D matrix (100, 2000)
    classes = sorted(ecdf_result["per_class"].keys())
    heatmap_data = np.array([ecdf_result["per_class"][c] for c in classes])

    plt.figure(figsize=(15, 8))
    
    # 2. Plot heatmap
    # cmap="YlOrRd": colors from yellow to red, red represents rank closer to 1.0 (more extreme)
    ax = sns.heatmap(heatmap_data, 
                     cmap="RdYlBu_r", 
                     center=0.5,
                     cbar_kws={'label': 'ECDF Rank Mean (1.0 = Extreme)'})

    plt.title(f"ECDF Extremeness Heatmap: {method_name}")
    plt.xlabel("Burst Index (0-2000)")
    plt.ylabel("Website Class (0-100)")
    # For observation convenience, you can limit to the first 500 bursts, as most perturbations are concentrated there
    # plt.xlim(0, 500) 
    
    plt.tight_layout()
    file_path=f'CenterInject_Explore/CenterInject_Watch/file_save/{method_name}.pdf'
    plt.savefig(file_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    
# plot_ecdf_heatmap(awa_ecdf,"AWA")
# plot_ecdf_heatmap(alert_ecdf, "Alert")
# plot_ecdf_heatmap(moe_ecdf, "Moe-Gen (Ours)")
# print("Heatmap saved as 'Defense_Methods_ECDF_Comparison.png'")