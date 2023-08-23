#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import threading

random.seed(24)


# Here I define the functions used to plot editor skills in each domain. 
# 
# - `plot_skill_hist` specifies the design properties of such histograms and allows the plotting of these histograms into a group of subplots.

# In[2]:


def plot_skill_hist(post_editors_table,ax,domain,colour,subplot_row,subplot_column):
    counts, edges, bars = ax[subplot_row,subplot_column].hist(post_editors_table[domain],bins=[0.5,1.5,2.5,3.5,4.5,5.5],edgecolor='black',color=colour)
    ax[subplot_row,subplot_column].set_title(domain.title(), fontsize=16)
    ax[subplot_row,subplot_column].set_xlabel(r'$S_{e}$', fontsize=12)
    ax[subplot_row,subplot_column].bar_label(bars)


# - The `editor_skills` function will give the absolute frequencies of the skills a given domain;

# In[3]:


def editor_skills(post_editors,domain):
    editor_skills_hist = np.histogram(post_editors[domain],bins=[0.5,1.5,2.5,3.5,4.5,5.5])
    editor_skills = editor_skills_hist[0]
    return editor_skills


# - The `quality` function calculates quality of a task, based on the skill of the editor and the probability $P(A)$ of editor to belong to a quality interval $A$, using the conditioned probabities $P(A|E)$ of belonging to the quality interval $A$, given that the task was assigned to editor $E$.

# In[ ]:


def quality(editor_skill, domain, Prob_A_given_E, post_editors_table):
    if (editor_skill>3):
        P_A_given_E = Prob_A_given_E[domain]['D']
    else:
        P_A_given_E = Prob_A_given_E[domain]['Not D']
    
    quality_interval_editor = random.choices(range(1,5),weights=P_A_given_E)[0]
    if quality_interval_editor ==1: return random.choice(range(1,26));
    if quality_interval_editor ==2: return random.choice(range(26,51));
    if quality_interval_editor ==3: return random.choice(range(51,76));
    if quality_interval_editor ==4: return random.choice(range(76,101));


# - The `Prob_A_plot` function plots the calculated $P(A)$ values for all domains;

# In[ ]:


def Prob_A_plot(Prob_A,ax,domain,colour,subplot_column):
    
    ax[subplot_column].bar(range(1,5),Prob_A[domain],color=colour)
    ax[subplot_column].set_title(domain.title(), fontsize=16)
    ax[subplot_column].set_xlim(0.5,4.5)
    ax[subplot_column].set_ylim(0,0.6)
    ax[subplot_column].set_xlabel('$A$',fontsize=12)
    
    if subplot_column == 0:
        ax[subplot_column].set_ylabel('$P(A)$', fontsize=12)
        ax[subplot_column].set_ylabel('$P(A)$', fontsize=12)


# - The `Prob_A_given_E_plot` function plots the calculated $P(A|E)$ values for all domains;

# In[ ]:


def Prob_A_given_E_plot(Prob_A_given_E,ax,domain,colour,subplot_column):
    
    ax[0,subplot_column].bar(range(1,5),Prob_A_given_E[domain]['D'],color=colour)
    ax[1,subplot_column].bar(range(1,5),Prob_A_given_E[domain]['Not D'],color=colour)
    ax[0,subplot_column].set_title(domain.title(), fontsize=16)
    ax[1,subplot_column].set_xlabel('$A$', fontsize=10)
    ax[0,subplot_column].set_xticks(range(1,5))
    ax[1,subplot_column].set_xticks(range(1,5))
    ax[0,subplot_column].set_ylim(0,0.8)
    ax[1,subplot_column].set_ylim(0,0.8)
    
    if subplot_column == 0:
        ax[0,subplot_column].set_ylabel('$P(A|E)$', fontsize=10)
        ax[1,subplot_column].set_ylabel('$P(A|E)$', fontsize=10)


# In[5]:


def editor_skill_for_task(task,post_editors_table, optimizing = False):
    if optimizing == False:
        return int(post_editors_table.loc[post_editors_table['id']==task['editor_assigned'],task['domain']])
    else:
        return int(post_editors_table.loc[post_editors_table['id']==task['new_editor'].to_list()[0],task['domain'].to_list()[0]])


# In[ ]:


def task_optimization(post_editors_table, tasks_table, number_task_swaps = 2e04,stringent=False):
    post_editors_table['sampling_weights'] = abs(post_editors_table['tasks_allocated']-post_editors_table['tasks_allocated'].mean())

    T_evaluations = [0]
    mean_quality = [tasks_table['quality_score'].mean()]
    stdev_quality = [tasks_table['quality_score'].std()]
    mean_skill = [tasks_table['editor_skill'].mean()]
    stdev_skill = [tasks_table['editor_skill'].std()]
    minimum_editor_tasks = [post_editors_table['tasks_allocated'].min()]
    assignment_evaluation = [objective_function(mean_skill[0],mean_skill,stdev_skill[0],stdev_skill,minimum_editor_tasks[0])]
    language_pairs = set(post_editors_table['language_pair'])
    T = 0
    
    for LP in language_pairs:
        T_lp = 0
        while T_lp <=number_task_swaps:
            T += 1
            T_lp += 1

            more_demanded_editor = post_editors_table[(post_editors_table['language_pair']==LP) & (post_editors_table['tasks_allocated']>300)].sample(n=1,weights='sampling_weights')
            sampled_task = tasks_table[tasks_table['editor_assigned'].isin(more_demanded_editor['id'])].sample(n=1)

            if T % 1000 == 0:
                print('Optimization in progress:','LP =',LP,', T =',T,
                  '| Mean quality =',"{0:.5f}".format(mean_quality[-1]),
                  '| Mean skill =',"{0:.5f}".format(mean_skill[-1]),
                  '| Quality STD =',"{0:.2f}".format(stdev_quality[-1]),
                  '| Skill STD =',"{0:.2f}".format(stdev_skill[-1]),
                  '| Minimum editor tasks =',minimum_editor_tasks[-1])

            if int(sampled_task['editor_skill'])==5:
                continue

            less_demanded_editor = post_editors_table[(post_editors_table['language_pair']==LP) & (post_editors_table['tasks_allocated']<62)].sample(n=1,weights='sampling_weights')
            sampled_task['new_editor'] = less_demanded_editor.iat[0,0]
            sampled_task['new_editor_skill'] = editor_skill_for_task(sampled_task,post_editors_table, optimizing=True)

            if int(sampled_task['new_editor_skill'])==1:
                continue

            sampled_task['quality_score'] = quality(sampled_task['new_editor_skill'].iloc[0],
                                                    sampled_task['domain'].iloc[0],
                                                    Prob_A_given_E, 
                                                    post_editors_table)
            new_mean_skill = float((tasks_table['editor_skill'].sum()-
                                    sampled_task['editor_skill']+sampled_task['new_editor_skill'])/len(tasks_table))
            new_stdev_skill = float(tasks_table['editor_skill'].agg(lambda skill: np.sqrt((sum((skill-new_mean_skill)**2)-
                            (sampled_task['editor_skill']-new_mean_skill)**2+
                            (sampled_task['new_editor_skill']-new_mean_skill)**2)/(len(tasks_table)-1))))
            new_minimum_tasks = min(post_editors_table['tasks_allocated'].min(),less_demanded_editor['tasks_allocated'].iloc[0])
            new_evaluation = objective_function(new_mean_skill,mean_skill[0],new_minimum_tasks)


            if int(sampled_task['new_editor_skill']) < int(sampled_task['editor_skill']):
                if stringent:
                    continue
                else:
                    metropolis_probability = metropolis(new_evaluation,assignment_evaluation[-1],T)
                    if random.random() > metropolis_probability:
                        continue

            tasks_table.at[sampled_task.index[0],'quality_score'] = sampled_task['quality_score']
            tasks_table.at[sampled_task.index[0],'editor_assigned'] = sampled_task['new_editor'].iloc[0]
            tasks_table.at[sampled_task.index[0],'editor_skill'] = sampled_task['new_editor_skill']

            post_editors_table.loc[post_editors_table['id']==more_demanded_editor['id'].to_list()[0],'tasks_allocated'] -=1 
            post_editors_table.loc[post_editors_table['id']==less_demanded_editor['id'].to_list()[0],'tasks_allocated'] +=1

            mean_quality.append(tasks_table['quality_score'].mean())
            stdev_quality.append(tasks_table['quality_score'].std())
            minimum_editor_tasks.append(new_minimum_tasks)
            mean_skill.append(new_mean_skill)
            stdev_skill.append(new_stdev_skill)
            assignment_evaluation.append(new_evaluation)
            T_evaluations.append(T)
    
    evaluation_df = pd.DataFrame({'T':T_evaluations, 'Minimum allocated tasks':minimum_editor_tasks,
                              'Mean skill': mean_skill, 'Mean quality': mean_quality,
                              'Skill STDEV': stdev_skill, 'Quality STDEV': stdev_quality, 'Evaluation': assignment_evaluation})
    return post_editors_table, tasks_table, evaluation_df


# In[ ]:


T = 0

def task_optimization_thr(post_editors_table,tasks_table,Prob_A_given_E,LP,mean_quality,mean_skill,stdev_quality,stdev_skill,minimum_editor_tasks,assignment_evaluation,T_evaluations,number_task_swaps=20000,stringent=False):
    global T
    vectors_lock = threading.Lock()
    T_lp = 0
    while T_lp <=number_task_swaps:
        
        T += 1
        Te = T
        T_lp += 1
        
        more_demanded_editor = post_editors_table[(post_editors_table['language_pair']==LP) & (post_editors_table['tasks_allocated']>300)].sample(n=1,weights='sampling_weights')
        sampled_task = tasks_table[tasks_table['editor_assigned'].isin(more_demanded_editor['id'])].sample(n=1)

        if Te % 1000 == 0:
            print('Optimization in progress:','LP =',LP,', T =',Te,
              '| Mean quality =',"{0:.5f}".format(mean_quality[-1]),
              '| Mean skill =',"{0:.5f}".format(mean_skill[-1]),
              '| Minimum editor tasks =',minimum_editor_tasks[-1])
        
        if int(sampled_task['editor_skill'])==5:
            continue
        
        less_demanded_editor = post_editors_table[(post_editors_table['language_pair']==LP) & (post_editors_table['tasks_allocated']<62)].sample(n=1,weights='sampling_weights')
        sampled_task['new_editor'] = less_demanded_editor.iat[0,0]
        sampled_task['new_editor_skill'] = editor_skill_for_task(sampled_task,post_editors_table, optimizing=True)
        
        if int(sampled_task['new_editor_skill'])==1:
            continue
            
        sampled_task['quality_score'] = quality(sampled_task['new_editor_skill'].iloc[0],
                                                sampled_task['domain'].iloc[0],
                                                Prob_A_given_E, 
                                                post_editors_table)
        new_mean_skill = float((tasks_table['editor_skill'].sum()-
                                sampled_task['editor_skill']+sampled_task['new_editor_skill'])/len(tasks_table))
        new_stdev_skill = float(tasks_table['editor_skill'].agg(lambda skill: np.sqrt((sum((skill-new_mean_skill)**2)-
                        (sampled_task['editor_skill']-new_mean_skill)**2+
                        (sampled_task['new_editor_skill']-new_mean_skill)**2)/(len(tasks_table)-1))))
        new_minimum_time = min(post_editors_table['tasks_allocated'].min(),less_demanded_editor['tasks_allocated'].iloc[0])
        new_evaluation = objective_function(new_mean_skill,mean_skill,new_minimum_time)
    
        
        if int(sampled_task['new_editor_skill']) < int(sampled_task['editor_skill']):
            if stringent:
                continue
            else:
                metropolis_probability = metropolis(new_evaluation,assignment_evaluation[-1],Te)
                if random.random() > metropolis_probability:
                    continue
        
        tasks_table.at[sampled_task.index[0],'quality_score'] = sampled_task['quality_score']
        tasks_table.at[sampled_task.index[0],'editor_assigned'] = sampled_task['new_editor'].iloc[0]
        tasks_table.at[sampled_task.index[0],'editor_skill'] = sampled_task['new_editor_skill']
    
        post_editors_table.loc[post_editors_table['id']==more_demanded_editor['id'].to_list()[0],'tasks_allocated'] -=1 
        post_editors_table.loc[post_editors_table['id']==less_demanded_editor['id'].to_list()[0],'tasks_allocated'] +=1
        
        vectors_lock.acquire()
        mean_quality.append(tasks_table['quality_score'].mean())
        stdev_quality.append(tasks_table['quality_score'].std())
        minimum_editor_tasks.append(post_editors_table['tasks_allocated'].min())
        mean_skill.append(new_mean_skill)
        stdev_skill.append(new_stdev_skill)
        assignment_evaluation.append(new_evaluation)
        T_evaluations.append(Te)
        vectors_lock.release()


# In[1]:


def objective_function(new_mean_skill,current_mean_skill,new_minimum_tasks):
    return (1-1e-6)*new_mean_skill/current_mean_skill+1e-6*new_minimum_tasks

def metropolis(new_evaluation,current_evaluation,T):
    t = 300 / T
    return np.exp(-5e5*(current_evaluation-new_evaluation)/t)


# In[ ]:


def first_assignment_plots(post_editors_table,tasks_table):
    fig_assign, (ax_assign1,ax_assign2,ax_assign3) = plt.subplots(1,3, figsize=(17,6))

    editor_skill_s = tasks_table['editor_skill'].value_counts().sort_index()

    ax_assign1.hist(post_editors_table['tasks_allocated'])
    ax_assign1.set_title('Tasks allocated per editor')
    ax_assign1.set_ylabel('Number of editors')
    ax_assign1.set_xlabel('Number of tasks allocated')
    bar_skill = ax_assign2.bar(editor_skill_s.index.to_list(),editor_skill_s.to_list())
    ax_assign2.set_title('Editor skill for task')
    ax_assign2.set_ylabel('Number of tasks')
    ax_assign2.set_xlabel('Editor skill, $S(E)$')
    ax_assign2.set_xticks(range(2,6))
    ax_assign2.set_xlim(1.5,5.5)
    ax_assign2.bar_label(bar_skill)
    ax_assign3.hist(tasks_table['quality_score'])
    ax_assign3.set_title('Task quality')
    ax_assign3.set_ylabel('Number of tasks')
    ax_assign3.set_xlabel('Task quality, $Q(t)$')
    plt.show()


# In[ ]:


def assignment_by_domain_plots(tasks_table):
    figd, ax_d = plt.subplots(2,3, figsize=(16,11))

    bard1 = ax_d[0,0].bar(tasks_table.groupby('domain')['editor_skill'].value_counts()['health_care'].sort_index().index,
                      tasks_table.groupby('domain')['editor_skill'].value_counts()['health_care'].sort_index(),color='magenta')
    ax_d[0,0].bar_label(bard1)
    ax_d[0,0].set_xlabel('Editor skill')
    ax_d[0,0].set_ylabel('Number of tasks')
    ax_d[0,0].set_xticks([2,3,4,5])
    ax_d[0,0].set_title('Healthcare')

    bard2 = ax_d[0,1].bar(tasks_table.groupby('domain')['editor_skill'].value_counts()['ecommerce'].sort_index().index,
                      tasks_table.groupby('domain')['editor_skill'].value_counts()['ecommerce'].sort_index(),color='green')
    ax_d[0,1].bar_label(bard2)
    ax_d[0,1].set_xlabel('Editor skill')
    ax_d[0,1].set_xticks([2,3,4,5])
    ax_d[0,1].set_title('Ecommerce')

    bard3 = ax_d[0,2].bar(tasks_table.groupby('domain')['editor_skill'].value_counts()['fintech'].sort_index().index,
                      tasks_table.groupby('domain')['editor_skill'].value_counts()['fintech'].sort_index(),color='red')
    ax_d[0,2].bar_label(bard3)
    ax_d[0,2].set_xlabel('Editor skill')
    ax_d[0,2].set_xticks([2,3,4,5])
    ax_d[0,2].set_title('Fintech')

    bard4 = ax_d[1,0].bar(tasks_table.groupby('domain')['editor_skill'].value_counts()['gamming'].sort_index().index,
                      tasks_table.groupby('domain')['editor_skill'].value_counts()['gamming'].sort_index(),color='yellow')
    ax_d[1,0].bar_label(bard4)
    ax_d[1,0].set_xlabel('Editor skill')
    ax_d[1,0].set_xticks([2,3,4,5])
    ax_d[1,0].set_title('Gaming')

    bard5 = ax_d[1,1].bar(tasks_table.groupby('domain')['editor_skill'].value_counts()['sports'].sort_index().index,
                      tasks_table.groupby('domain')['editor_skill'].value_counts()['sports'].sort_index(),color='purple')
    ax_d[1,1].bar_label(bard5)
    ax_d[1,1].set_xlabel('Editor skill')
    ax_d[1,1].set_xticks([2,3,4,5])
    ax_d[1,1].set_title('Sports')

    bard6 = ax_d[1,2].bar(tasks_table.groupby('domain')['editor_skill'].value_counts()['travel'].sort_index().index,
                      tasks_table.groupby('domain')['editor_skill'].value_counts()['travel'].sort_index(),color='brown')
    ax_d[1,2].bar_label(bard6)
    ax_d[1,2].set_xlabel('Editor skill')
    ax_d[1,2].set_xticks([2,3,4,5])
    ax_d[1,2].set_xlim(1.5,5.5)
    ax_d[1,2].set_title('Travel')
    plt.show()


# In[ ]:


def optimization_plots(evaluation_df):
    fig_opti,ax_opti = plt.subplots(2,3, figsize=(18,16))
    ax_opti[0,0].plot(evaluation_df['T'],evaluation_df['Mean skill'])
    ax_opti[0,0].set_title('Task skill during optimization')
    ax_opti[0,0].set_ylabel('Mean skill for task')
    ax_opti[0,0].set_xlabel('T')
    ax_opti[0,1].plot(evaluation_df['T'],evaluation_df['Mean quality'])
    ax_opti[0,1].set_title('Task quality during optimization')
    ax_opti[0,1].set_ylabel('Mean task quality')
    ax_opti[0,1].set_xlabel('T')
    ax_opti[1,0].plot(evaluation_df['T'],evaluation_df['Skill STDEV'])
    ax_opti[1,0].set_title('Task skill variation during optimization')
    ax_opti[1,0].set_ylabel('Skill standard deviation for task')
    ax_opti[1,0].set_xlabel('T')
    ax_opti[1,1].plot(evaluation_df['T'],evaluation_df['Quality STDEV'])
    ax_opti[1,1].set_title('Task quality variation during optimization')
    ax_opti[1,1].set_ylabel('Task quality standard deviation')
    ax_opti[1,1].set_xlabel('T')
    ax_opti[0,2].plot(evaluation_df['T'],evaluation_df['Minimum allocated tasks'])
    ax_opti[0,2].set_title('Minimum allocated tasks during optimization')
    ax_opti[0,2].set_ylabel('Minimum allocated tasks to an editor')
    ax_opti[0,2].set_xlabel('T')
    ax_opti[1,2].plot(evaluation_df['T'],evaluation_df['Evaluation'])
    ax_opti[1,2].set_title('Evaluation during optimization')
    ax_opti[1,2].set_ylabel('Evaluation')
    ax_opti[1,2].set_xlabel('T')
    plt.show()

