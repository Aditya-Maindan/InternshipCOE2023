import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency

df = pd.read_csv('/Users/adityamaindan/Desktop/internship proj/instagram.csv')

numeric_df = df.select_dtypes(include=['number'])
correlation_matrix = numeric_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()

plt.figure(figsize=(12, 8))  # Adjust the figure size as needed

# Loop through each variable and create a box plot
for col in df.columns:
    plt.subplot(3, 4, df.columns.get_loc(col) + 1)  # Adjust the subplot layout as needed
    sns.boxplot(data=df, y=col)
    plt.title(col)
    plt.xlabel('')
    plt.ylabel('')

plt.tight_layout()
plt.show()

# # Create subplots
# sns.set(style="whitegrid")

# # Create a box plot to visualize the relationship
# plt.figure(figsize=(8, 6))
# sns.boxplot(x='fake', y='profile pic', data=df)
# plt.title('Box Plot of profile pic vs. fake')
# plt.show()

# plt.figure(figsize=(8, 6))
# sns.boxplot(x='fake', y='nums/length username', data=df)
# plt.title('Box Plot of nums/length username vs. fake')
# plt.show()

# plt.figure(figsize=(8, 6))
# sns.boxplot(x='fake', y='fullname words', data=df)
# plt.title('Box Plot of fullname words vs. fake')
# plt.show()

# plt.figure(figsize=(8, 6))
# sns.boxplot(x='fake', y='nums/length fullname', data=df)
# plt.title('Box Plot of nums/length fullname vs. fake')
# plt.show()

# plt.figure(figsize=(8, 6))
# sns.boxplot(x='fake', y='name==username', data=df)
# plt.title('Box Plot of name==username vs. fake')  ///////remove private
# plt.show()

# plt.figure(figsize=(8, 6))
# sns.boxplot(x='fake', y='description length', data=df)
# plt.title('Box Plot of description length vs. fake')
# plt.show()

# plt.figure(figsize=(8, 6))
# sns.boxplot(x='fake', y='external URL', data=df)
# plt.title('Box Plot of external URL vs. fake')
# plt.show()

# plt.figure(figsize=(8, 6))
# sns.boxplot(x='fake', y='profile pic', data=df)
# plt.title('Box Plot of profile pic vs. fake')
# plt.show()

# plt.figure(figsize=(8, 6))
# sns.boxplot(x='fake', y='private', data=df)
# plt.title('Box Plot of private vs. fake')
# plt.show()

# plt.figure(figsize=(8, 6))
# sns.boxplot(x='fake', y='#posts', data=df)
# plt.title('Box Plot of #posts vs. fake')
# plt.show()

# plt.figure(figsize=(8, 6))
# sns.boxplot(x='fake', y='#followers', data=df)
# plt.title('Box Plot of #followers vs. fake')
# plt.show()

# plt.figure(figsize=(8, 6))
# sns.boxplot(x='fake', y='#follows', data=df)
# plt.title('Box Plot of #follows vs. fake')
# plt.show()


# print("\n\n Ttest for profile pic and fake:")
# group_A = df['profile pic']
# group_B = df['fake']

# t_stat, p_value = stats.ttest_1samp(group_A, np.mean(group_B))

# # Display the results
# print(f"t-statistic: {t_stat}")
# print(f"P-value: {p_value}")

# # Determine statistical significance
# alpha = 0.05  # Set your desired alpha level
# if p_value < alpha:
#     print("Reject the null hypothesis: There is a significant difference between the groups.")
# else:
#     print("Fail to reject the null hypothesis: There is no significant difference between the groups.")
    
# print("\n\n Ttest for fullname words and fake:")
# group_A = df['fullname words']
# group_B = df['fake']

# t_stat, p_value = stats.ttest_1samp(group_A, np.mean(group_B))

# # Display the results
# print(f"t-statistic: {t_stat}")
# print(f"P-value: {p_value}")

# # Determine statistical significance
# alpha = 0.05  # Set your desired alpha level
# if p_value < alpha:
#     print("Reject the null hypothesis: There is a significant difference between the groups.")
# else:
#     print("Fail to reject the null hypothesis: There is no significant difference between the groups.")   
    

# print("\n\n Ttest for nums/length fullname and fake:")
# group_A = df['nums/length fullname']
# group_B = df['fake']

# t_stat, p_value = stats.ttest_1samp(group_A, np.mean(group_B))

# # Display the results
# print(f"t-statistic: {t_stat}")
# print(f"P-value: {p_value}")

# # Determine statistical significance
# alpha = 0.05  # Set your desired alpha level
# if p_value < alpha:
#     print("Reject the null hypothesis: There is a significant difference between the groups.")
# else:
#     print("Fail to reject the null hypothesis: There is no significant difference between the groups.")  
    
# print("\n\n Ttest for name==username and fake:")
# group_A = df['name==username']
# group_B = df['fake']

# t_stat, p_value = stats.ttest_1samp(group_A, np.mean(group_B))

# # Display the results
# print(f"t-statistic: {t_stat}")
# print(f"P-value: {p_value}")

# # Determine statistical significance
# alpha = 0.05  # Set your desired alpha level
# if p_value < alpha:
#     print("Reject the null hypothesis: There is a significant difference between the groups.")
# else:
#     print("Fail to reject the null hypothesis: There is no significant difference between the groups.") 
    
# print("\n\n Ttest for description length and fake:")
# group_A = df['description length']
# group_B = df['fake']

# t_stat, p_value = stats.ttest_1samp(group_A, np.mean(group_B))

# # Display the results
# print(f"t-statistic: {t_stat}")
# print(f"P-value: {p_value}")

# # Determine statistical significance
# alpha = 0.05  # Set your desired alpha level
# if p_value < alpha:
#     print("Reject the null hypothesis: There is a significant difference between the groups.")
# else:
#     print("Fail to reject the null hypothesis: There is no significant difference between the groups.") 
    
# print("\n\n Ttest for #posts and fake:")
# group_A = df['#posts']
# group_B = df['fake']

# t_stat, p_value = stats.ttest_1samp(group_A, np.mean(group_B))

# # Display the results
# print(f"t-statistic: {t_stat}")
# print(f"P-value: {p_value}")

# # Determine statistical significance
# alpha = 0.05  # Set your desired alpha level
# if p_value < alpha:
#     print("Reject the null hypothesis: There is a significant difference between the groups.")
# else:
#     print("Fail to reject the null hypothesis: There is no significant difference between the groups.")       

    
# print("\n\n Ttest for #followers and fake:")
# group_A = df['#followers']
# group_B = df['fake']

# t_stat, p_value = stats.ttest_1samp(group_A, np.mean(group_B))

# # Display the results
# print(f"t-statistic: {t_stat}")
# print(f"P-value: {p_value}")

# # Determine statistical significance
# alpha = 0.05  # Set your desired alpha level
# if p_value < alpha:
#     print("Reject the null hypothesis: There is a significant difference between the groups.")
# else:
#     print("Fail to reject the null hypothesis: There is no significant difference between the groups.") 
    
# print("\n\n Ttest for #follows and fake:")
# group_A = df['#follows']
# group_B = df['fake']

# t_stat, p_value = stats.ttest_1samp(group_A, np.mean(group_B))

# # Display the results
# print(f"t-statistic: {t_stat}")
# print(f"P-value: {p_value}")

# # Determine statistical significance
# alpha = 0.05  # Set your desired alpha level
# if p_value < alpha:
#     print("Reject the null hypothesis: There is a significant difference between the groups.")
# else:
#     print("Fail to reject the null hypothesis: There is no significant difference between the groups.")   
    
# print("\n\n Chi-Square Test for profile pic and fake:") 
# contingency_table = pd.crosstab(df['profile pic'], df['fake'])

# # Perform a chi-squared test
# chi2, p, dof, expected = chi2_contingency(contingency_table)

# # Display the results
# print(f"Chi-squared statistic: {chi2}")
# print(f"P-value: {p}")

# # Determine statistical significance
# alpha = 0.05  # Set your desired alpha level
# if p < alpha:
#     print("Reject the null hypothesis: There is a significant association between Independent_Variable and Dependent_Variable.")
# else:
#     print("Fail to reject the null hypothesis: There is no significant association between Independent_Variable and Dependent_Variable.")
    
# cross_tab = pd.crosstab(df['profile pic'], df['fake'])

# # Display the crosstab
# print(cross_tab)

# print("\n\n Chi-Square Test for fullname words and fake:") 
# contingency_table = pd.crosstab(df['fullname words'], df['fake'])

# # Perform a chi-squared test
# chi2, p, dof, expected = chi2_contingency(contingency_table)

# # Display the results
# print(f"Chi-squared statistic: {chi2}")
# print(f"P-value: {p}")

# # Determine statistical significance
# alpha = 0.05  # Set your desired alpha level
# if p < alpha:
#     print("Reject the null hypothesis: There is a significant association between Independent_Variable and Dependent_Variable.")
# else:
#     print("Fail to reject the null hypothesis: There is no significant association between Independent_Variable and Dependent_Variable.") 
    
# print("\n\n Chi-Square Test for nums/length fullname and fake:") 
# contingency_table = pd.crosstab(df['nums/length fullname'], df['fake'])

# # Perform a chi-squared test
# chi2, p, dof, expected = chi2_contingency(contingency_table)

# # Display the results
# print(f"Chi-squared statistic: {chi2}")
# print(f"P-value: {p}")

# # Determine statistical significance
# alpha = 0.05  # Set your desired alpha level
# if p < alpha:
#     print("Reject the null hypothesis: There is a significant association between Independent_Variable and Dependent_Variable.")
# else:
#     print("Fail to reject the null hypothesis: There is no significant association between Independent_Variable and Dependent_Variable.") 
    
# print("\n\n Chi-Square Test for description length and fake:") 
# contingency_table = pd.crosstab(df['description length'], df['fake'])

# # Perform a chi-squared test
# chi2, p, dof, expected = chi2_contingency(contingency_table)

# # Display the results
# print(f"Chi-squared statistic: {chi2}")
# print(f"P-value: {p}")

# # Determine statistical significance
# alpha = 0.05  # Set your desired alpha level
# if p < alpha:
#     print("Reject the null hypothesis: There is a significant association between Independent_Variable and Dependent_Variable.")
# else:
#     print("Fail to reject the null hypothesis: There is no significant association between Independent_Variable and Dependent_Variable.") 
    
# print("\n\n Chi-Square Test for #posts and fake:") 
# contingency_table = pd.crosstab(df['#posts'], df['fake'])

# # Perform a chi-squared test
# chi2, p, dof, expected = chi2_contingency(contingency_table)

# # Display the results
# print(f"Chi-squared statistic: {chi2}")
# print(f"P-value: {p}")

# # Determine statistical significance
# alpha = 0.05  # Set your desired alpha level
# if p < alpha:
#     print("Reject the null hypothesis: There is a significant association between Independent_Variable and Dependent_Variable.")
# else:
#     print("Fail to reject the null hypothesis: There is no significant association between Independent_Variable and Dependent_Variable.") 
    
# print("\n\n Chi-Square Test for #follows and fake:") 
# contingency_table = pd.crosstab(df['#follows'], df['fake'])

# # Perform a chi-squared test
# chi2, p, dof, expected = chi2_contingency(contingency_table)

# # Display the results
# print(f"Chi-squared statistic: {chi2}")
# print(f"P-value: {p}")

# # Determine statistical significance
# alpha = 0.05  # Set your desired alpha level
# if p < alpha:
#     print("Reject the null hypothesis: There is a significant association between Independent_Variable and Dependent_Variable.")
# else:
#     print("Fail to reject the null hypothesis: There is no significant association between Independent_Variable and Dependent_Variable.")            
        
# print("\n\n Chi-Square Test for #followers and fake:") 
# contingency_table = pd.crosstab(df['#followers'], df['fake'])

# # Perform a chi-squared test
# chi2, p, dof, expected = chi2_contingency(contingency_table)

# # Display the results
# print(f"Chi-squared statistic: {chi2}")
# print(f"P-value: {p}")

# # Determine statistical significance
# alpha = 0.05  # Set your desired alpha level
# if p < alpha:
#     print("Reject the null hypothesis: There is a significant association between Independent_Variable and Dependent_Variable.")
# else:
#     print("Fail to reject the null hypothesis: There is no significant association between Independent_Variable and Dependent_Variable.")            
                               
             
        
        
             


