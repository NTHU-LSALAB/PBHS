from functools import total_ordering
import random
from typing import Dict, Optional
import logging
import time
import numpy as np
import math
import time
import scipy.stats as stats
import ray
from ray.tune import result
from ray.tune.schedulers.trial_scheduler import FIFOScheduler, TrialScheduler
from ray.tune.result import DEFAULT_METRIC
from ray.tune import trial_runner
from ray.tune.trial import Trial
from ray.tune.resources import Resources
from ray.tune.utils.trainable import TrainableUtil
from ray.tune.schedulers.pbhs_assistant.readJson import *
# from ray.tune.schedulers.pbhs_assistant.LCNet import *
from ray.tune.schedulers.pbhs_assistant.lc_extrapolation import *
from ray.tune.schedulers.pbhs_assistant.scipy_fit import *
import sys

# sys.path.append("/home/didwdidw/project/pybnn")

# from pybnn.lcnet import LCNet


logger = logging.getLogger(__name__)

class tracker:
    def __init__(self):
        self.trial_accuracy_records = {}
    
    def get_trial_accuracy_records(self, trial_id) -> list:
        if(trial_id in self.trial_accuracy_records):
            return self.trial_accuracy_records[trial_id]

    def set_trial_record(self, trial_id, accuracy):
        if(trial_id not in self.trial_accuracy_records):
            self.trial_accuracy_records[trial_id] = []
            self.trial_accuracy_records[trial_id].append(accuracy)
        else:
            self.trial_accuracy_records[trial_id].append(accuracy)

### Prediction-based Scheduler for Hyperparameter Tuning
class PBHS(FIFOScheduler):
    def __init__(self, gpus_limit, metric_key, time_limit, max_iteration, exploration_ratio, dynamic_exploration, reliable_threshold=0.1, check_n_prediction_records=2):
        print("Launching PBHS")
        FIFOScheduler.__init__(self)
        self.metric_key = metric_key
        # self._mode = mode
        
        self.sorted_running_trials = [] # [[trial_id, predictive_mean, standard_deviation]]      
        self.unpromising_trials = []
        self.sorted_suspended_trials = []
        self.scheduling_interval = None
        self.current_accuracy = {}
        self.best_trial_info = [None, 0, 0] # [trial_id, iteration, accuracy]
        
        self.last_trials_info = []
        self.last_stage_launcher = False
        self.last_stage = False
        self.D = reliable_threshold
        self.reliability_test_interval = 1
        self.check_n_prediction_records = check_n_prediction_records

        self.convergence_iteration = max_iteration
        self.total_gpus = int(ray.cluster_resources().get("GPU", 0)) if not gpus_limit else gpus_limit
        self.dynamic_exploration_ratio = dynamic_exploration
        self.exploration_ratio = exploration_ratio
        self.min_exporation_ratio = 0.0
        #self.prediction_model = LCNet()
        self.initial_tolerance = 0.95
        self.min_tolerance = 0.95
        self.max_tolerance = 0.95
        self.tolerance = self.initial_tolerance

        self.result_paths = dict()
        self.experiment_time_limit = time_limit
        self.single_iteration_times = []
        self.prediction_interval = 10
        self.prediction_checkpoint = [4, 8, 16, 32, 64, 128, 256, 512]
        self.reliability_checkpoint = [x for x in range(4, 500, self.reliability_test_interval)]
        self.max_iteration = max_iteration
        self.experiment_start_time = time.time()
        self.experiment_current_time = time.time()
        self.experiment_phase1_time_limit = None
        self.experiment_elapsed_time = 0
        self.total_prediction_time = 0
        self.prediction_history = dict()
        self.reliable_prediction = dict()
        self.predictive_near_future_accuracy = dict()
        self.total_prediction_error = [0, 0, 0]
        self.completed_trials = 0

    # def train_prediction_model(self, numOfTrials, currentIteration, totalIteration, trial_id, newAcc):
    #     PATH_TO_RESULTS = []
    #     for i in range(numOfTrials):
    #         PATH_TO_RESULTS.append( self.result_paths[ trial_id[i] ] )
    #     configs = readConfig(numOfTrials, PATH_TO_RESULTS)
    #     if (currentIteration > 1):
    #         acc = readAcc(numOfTrials, currentIteration - 1, PATH_TO_RESULTS)
    #         for i in range(numOfTrials):
    #             acc[i].append( newAcc[i] )
    #     else:
    #         acc = []
    #         for i in range(numOfTrials):
    #             acc.append( [newAcc[i]] )
        
    #     train(self.prediction_model, currentIteration, totalIteration, numOfTrials, configs, acc)
    def scoring(self, pred_weight, acc_weight, pred_mean, acc):
        score = pred_weight * pred_mean + acc_weight * acc
        return score
    
    def validate_mean(self, runner, mean, trial_id):
        if(mean == 0.001 or mean > 3.0):
            self.update_unpromising_trial(trial_id)
            self.remove_unpromising_trials_from_candidate_pool(runner)
        
    def predict(self, currentIteration, totalIteration, trial_id, newAcc):
        PATH_TO_RESULTS = self.result_paths[trial_id]
        # configs = readConfig(1, PATH_TO_RESULTS)
        # result = predict(self.prediction_model, 1, configs)
        # result[1][0] = math.sqrt(result[1][0])
        if (currentIteration > 1):
            acc = readSingleAcc (currentIteration - 1, PATH_TO_RESULTS, self.metric_key)
            acc.append(newAcc)
        else:
            acc = []
            acc.append(newAcc)

        result = extrapolate (currentIteration, totalIteration, acc)
        # result = scipy_fit(currentIteration, totalIteration, acc)
        return result  # result[0] = mean, result[1] = std

    def takeAccuracy(self, elem):
        return elem[1]

    def update_prediction_history(self, trial_id, predictive_mean, standard_deviation):
        self.prediction_history[trial_id].append([predictive_mean, standard_deviation])

    def update_sorted_running_trial(self, trial_id, predictive_mean, standard_deviation):
        if not self.sorted_running_trials:
            self.sorted_running_trials.append([trial_id, predictive_mean, standard_deviation])
        else:
            if(self.exist(trial_id, self.sorted_running_trials)):
                target_index = self.find_trial_index(trial_id, self.sorted_running_trials)
                self.sorted_running_trials[target_index][1] = predictive_mean
                self.sorted_running_trials[target_index][2] = standard_deviation
                self.sorted_running_trials.sort(key = self.takeAccuracy, reverse = True)
                return
            if(len(self.sorted_running_trials) < self.total_gpus):
                for i in range(0, len(self.sorted_running_trials)):
                    if(predictive_mean > self.sorted_running_trials[i][1]):
                        self.sorted_running_trials.insert(i, [trial_id, predictive_mean, standard_deviation])
                        return
                self.sorted_running_trials.append([trial_id, predictive_mean, standard_deviation])
            else:
                for i in range(0, len(self.sorted_running_trials)):
                    if(predictive_mean >= self.sorted_running_trials[i][1]):
                        self.sorted_running_trials.insert(i, [trial_id, predictive_mean, standard_deviation])
                        paused_trial_info = self.sorted_running_trials.pop()
                        self.update_sorted_suspended_trial(paused_trial_info[0], paused_trial_info[1], paused_trial_info[2])
                        return
                self.sorted_running_trials.append([trial_id, predictive_mean, standard_deviation])

    def update_sorted_suspended_trial(self, trial_id, predictive_mean, standard_deviation):
        if not self.sorted_suspended_trials:
            self.sorted_suspended_trials.append([trial_id, predictive_mean, standard_deviation])
        else:
            if(self.exist(trial_id, self.sorted_suspended_trials)):
                target_index = self.find_trial_index(trial_id, self.sorted_suspended_trials)
                self.sorted_suspended_trials[target_index][1] = predictive_mean
                self.sorted_suspended_trials[target_index][2] = standard_deviation
                self.sorted_suspended_trials.sort(key = self.takeAccuracy, reverse = True)
                return
            for i in range(0, len(self.sorted_suspended_trials)):
                if(predictive_mean > self.sorted_suspended_trials[i][1]):
                    self.sorted_suspended_trials.insert(i, [trial_id, predictive_mean, standard_deviation])
                    return
            self.sorted_suspended_trials.append([trial_id, predictive_mean, standard_deviation])
                
    def update_unpromising_trial(self, trial_id):
        if(trial_id not in self.unpromising_trials):
            self.unpromising_trials.append(trial_id)

    def greater(self, trial_info0, trial_info1):
        if trial_info0[1] > trial_info1[1]:
            return True

    def predict_convergence_accuracy(self, current_iteration, trial_id, newAcc):
        result = self.predict(current_iteration, self.convergence_iteration, trial_id, newAcc)
        mean = result[0]
        standard_deviation = result[1]
        return mean, standard_deviation

    def predict_near_future_accuracy(self, current_iteration, interval, trial_id, newAcc):
        current_checkpoint = self.reliability_checkpoint.index(current_iteration)
        result = self.predict(current_iteration, self.reliability_checkpoint[current_checkpoint + interval], trial_id, newAcc)
        mean = result[0]
        standard_deviation = result[1]
        return mean, standard_deviation
    
    def show_comparison_info(self, worse_trial_info, trial0_info, trial1_info, prob):
        print(trial0_info, trial1_info, "probability:", prob)
        print("confirm worse trial:", worse_trial_info[0])

    def confirm_worse_trial(self, trial0_info, trial1_info):
        if((trial0_info[1] == 0.0 and trial0_info[2] == 0.0) or (trial1_info[1] == 0.0 and trial1_info[2] == 0.0)):
            return -1, 0.0
        mean0, std0 = trial0_info[1], trial0_info[2]
        mean1, std1 = trial1_info[1], trial1_info[2]
        new_distribution_mean = mean0 - mean1
        new_distribution_std = (std0**2 + std1**2)**0.5

        ###############################################################################
        ### calculate the probability of trial0 being less than trial1
        ###############################################################################
        probability = stats.norm.cdf(0, loc=new_distribution_mean, scale=new_distribution_std)
        if(probability > 0.5):
            if(probability > self.tolerance):
                return trial0_info[0], probability
            else:
                return -1, 0.0
        elif(probability < 0.5):
            # probability of trial1 being less than trial0
            probability = 1 - probability
            if(probability > self.tolerance):
                return trial1_info[0], probability
            else:
                return -1, 0.0
        return -1, 0.0

    ####################################################
    ### Get all candidates of the candidate pool.
    ####################################################
    def get_all_candidates(self):
        exploitation_running_trials = []
        for trial_info in self.sorted_running_trials:
            if(trial_info[1] != 0.0 and trial_info[2] != 0.0):
                exploitation_running_trials.append(trial_info)
        all_candidates = exploitation_running_trials + self.sorted_suspended_trials
        all_candidates.sort(key = self.takeAccuracy, reverse = True) 
        return all_candidates

    def remove_unpromising_trials_from_candidate_pool(self, runner):
        temp_running_list = [item for item in self.sorted_running_trials if item[0] not in self.unpromising_trials]
        temp_suspended_list = [item for item in self.sorted_suspended_trials if item[0] not in self.unpromising_trials]
        # temp_prediction_history_list = [item for item in self.prediction_history if item[0] not in self.unpromising_trials]
        self.sorted_running_trials = temp_running_list
        self.sorted_suspended_trials = temp_suspended_list
        # self.prediction_history = temp_prediction_history_list
        for trial in runner.get_trials():
            if (trial.status == Trial.PAUSED and (trial.trial_id in self.unpromising_trials)):
                runner.stop_trial(trial)

    def on_complete(self, trial_id):
        pass
        
    def kill_all_unpromising_trials(self, runner):
        all_candidates = self.get_all_candidates()
        for i in range(0, len(all_candidates)):
            for trial_id in self.prediction_history:
                probs = []
                if(len(self.prediction_history[trial_id]) < self.check_n_prediction_records):
                    continue
                should_kill_current_candidate = True
                for j in range(1, self.check_n_prediction_records + 1):
                    pred_record = [trial_id, self.prediction_history[trial_id][-j][0], self.prediction_history[trial_id][-j][1]]
                    worse_trial_id, prob = self.confirm_worse_trial(all_candidates[i], pred_record)
                    probs.append(prob)
                    if(worse_trial_id != all_candidates[i][0]):
                        should_kill_current_candidate = False
                        break
                if(should_kill_current_candidate):
                    for j in range(1, self.check_n_prediction_records + 1):
                        print(self.prediction_history)
                        pred_record = [trial_id, self.prediction_history[trial_id][-j][0], self.prediction_history[trial_id][-j][1]]
                        self.show_comparison_info(all_candidates[i], all_candidates[i], pred_record, probs)
                    self.update_unpromising_trial(worse_trial_id)
        self.remove_unpromising_trials_from_candidate_pool(runner)

    def prioritize_all_candidates(self):      
        all_candidates = self.get_all_candidates()
        for i in range(0, self.n_exploitation_running_trials):
            self.sorted_running_trials[i] = all_candidates[i]
        if(len(all_candidates) > self.n_exploitation_running_trials):
            self.sorted_suspended_trials = all_candidates[self.n_exploitation_running_trials: ]
        else:
            self.sorted_suspended_trials = []
  
    def maintain_exploration_ratio(self, runner):
        self.kill_all_unpromising_trials(runner)
        for i in range(0, len(self.sorted_running_trials) - 1):
            if(self.n_exploitation_running_trials > self.expected_exploitative_gpus):
                trial_info = self.sorted_running_trials.pop(self.n_exploitation_running_trials - 1)
                self.update_sorted_suspended_trial(trial_info[0], trial_info[1], trial_info[2])

    def exist(self, target, myList):
        for item in myList:
            if target in item:
                return True
        return False
    
    def find_trial_index(self, target, myList):
        for i in range(0, len(myList)):
            if target in myList[i]:
                return i
        raise IndexError

    def update_experiment_status(self):
        self.experiment_current_time = time.time()
        self.experiment_elapsed_time = self.experiment_current_time - self.experiment_start_time
        self.experiment_phase1_time_limit = self.experiment_time_limit - self.expected_trial_duration
        # self.tolerance = max(self.initial_tolerance - (self.initial_tolerance - self.min_tolerance) * (self.experiment_elapsed_time / self.experiment_phase1_time_limit), self.min_tolerance)
        self.tolerance = min(self.initial_tolerance + (self.max_tolerance - self.initial_tolerance) * (self.experiment_elapsed_time / self.experiment_phase1_time_limit), self.max_tolerance)
        if(self.dynamic_exploration_ratio):
            self.exploration_ratio = max((self.experiment_phase1_time_limit - self.experiment_elapsed_time) / self.experiment_phase1_time_limit, self.min_exporation_ratio)

    def print_pbhs_info(self, position):
        print("=======================PBHS info==============================")
        print(position)
        print("Total GPUs: ", self.total_gpus)
        print("Available GPUs: ", ray.available_resources().get("GPU", 0))
        print("Sorted running trials: ", self.sorted_running_trials)
        print("Sorted suspended trials: ", self.sorted_suspended_trials)
        print("unpromising trials: ", self.unpromising_trials)
        print("Experiment time limit: ", self.experiment_time_limit)
        print("Elapsed time: ", self.experiment_elapsed_time)
        print("Phase1 time limit: ", self.experiment_phase1_time_limit)
        print("Total prediction time: ", self.total_prediction_time)
        print("Dynamic exploration: ", self.dynamic_exploration_ratio)
        print("Exploration ratio: ", self.exploration_ratio)
        print("#GPUs in exploration: ", self.expected_explorative_gpus)
        print("#GPUs in exploitation: ", self.expected_exploitative_gpus)
        print("Tolerance: ", self.tolerance)
        print("Last stage trials info:", self.last_trials_info)
        print("Reliable prediction list: ", self.reliable_prediction)
        print("Near future prediction: ", self.predictive_near_future_accuracy)
        print("Prediction history: ", self.prediction_history)
        print("==============================================================")
    
    def schedule(self, runner):
        self.print_pbhs_info("Before scheduling")
        self.prioritize_all_candidates()
        self.maintain_exploration_ratio(runner)
        self.print_pbhs_info("After scheduling")

    @property
    def metric(self):
        return self._metric

    def set_search_properties(self, metric: Optional[str],
                              mode: Optional[str]) -> bool:
        """Pass search properties to scheduler.

        This method acts as an alternative to instantiating schedulers
        that react to metrics with their own `metric` and `mode` parameters.

        Args:
            metric (str): Metric to optimize
            mode (str): One of ["min", "max"]. Direction to optimize.
        """
        
        if self._metric and metric:
            return False
        if metric:
            self._metric = metric

        if self._metric is None:
            # Per default, use anonymous metric
            self._metric = DEFAULT_METRIC

        return True
        
    def resource_available(self, trial_runner: "trial_runner.TrialRunner") -> bool:
        if trial_runner.has_resources(self.pendingTrials[-1].resources):
            return True
        else:
            return False

    def on_trial_add(self, trial_runner: "trial_runner.TrialRunner",
                     trial: Trial):
        print("Add new trial:", trial.trial_id)
        self.prediction_history[trial.trial_id] = list()
        # if(resource_available(trial_runner, pendingTrials[-1])):
        #     add_trial()
    
    def on_trial_error(self, trial_runner: "trial_runner.TrialRunner",
                       trial: Trial):
        print("on_trial_error")
    
    def on_trial_result(self, trial_runner: "trial_runner.TrialRunner",
                        trial: Trial, result: Dict) -> str:
        print('trial result:', result)
        print("$$: ", trial.trial_id, trial.status)
        
        # train_prediction_model(self, numOfTrials, currentIteration, totalIteration, trial_id, newAcc)
        self.result_paths[trial.trial_id] = trial.logdir + '/result.json'
        current_iteration = result['training_iteration']
        current_accuracy = result[self.metric_key]
        current_trial_id = [trial.trial_id]
        mean = 0.0
        if(trial.trial_id not in self.reliable_prediction):
            self.reliable_prediction[trial.trial_id] = False

        if(trial.trial_id in self.unpromising_trials): 
            return TrialScheduler.STOP

        if(not self.last_stage):
            self.single_iteration_times += [result["time_this_iter_s"]]
            # self.expected_trial_duration = min(self.expected_trial_duration, self.max_iteration * result['time_this_iter_s'])
            # self.longest_trial_duration = max(self.longest_trial_duration, result['time_total_s'])
            # self.expected_trial_duration = max(self.expected_trial_duration, self.longest_trial_duration)
            # print("expected trial duration:", self.expected_trial_duration)
            self.update_experiment_status()

            if(current_iteration >= 4 
                and trial.trial_id in self.reliable_prediction and not self.reliable_prediction[trial.trial_id] 
                and current_iteration in self.reliability_checkpoint):

                prediction_start_time = time.time()
                if(trial.trial_id not in self.predictive_near_future_accuracy):
                    print("start predict near future ", trial.trial_id)
                    mean, standard_deviation = self.predict_near_future_accuracy(current_iteration, 1, trial.trial_id, current_accuracy)
                    print("end predict near future ", trial.trial_id)
                    self.validate_mean(trial_runner, mean, trial.trial_id)
                    self.predictive_near_future_accuracy[trial.trial_id] = mean

                elif(not self.reliable_prediction[trial.trial_id]):
                    if(abs(self.predictive_near_future_accuracy[trial.trial_id] - current_accuracy) < self.D):
                        self.reliable_prediction[trial.trial_id] = True
                        self.predictive_near_future_accuracy[trial.trial_id] = -1.0
                        print("start predict convergence1 ", trial.trial_id)
                        mean, standard_deviation = self.predict_convergence_accuracy(current_iteration, trial.trial_id, current_accuracy)
                        print("end predict convergence1 ", trial.trial_id)
                        self.validate_mean(trial_runner, mean, trial.trial_id)
                        self.update_prediction_history(trial.trial_id, mean, standard_deviation)

                        if(self.exist(trial.trial_id, self.sorted_running_trials)):
                            self.update_sorted_running_trial(trial.trial_id, mean, standard_deviation)
                        elif(self.exist(trial.trial_id, self.sorted_suspended_trials)):
                            self.update_sorted_suspended_trial(trial.trial_id, mean, standard_deviation)

                    else:
                        print("start predict near future1 ", trial.trial_id)
                        mean, standard_deviation = self.predict_near_future_accuracy(current_iteration, 1, trial.trial_id, current_accuracy)
                        print("start predict near future1 ", trial.trial_id)
                        self.validate_mean(trial_runner, mean, trial.trial_id)
                        self.predictive_near_future_accuracy[trial.trial_id] = mean
            
                prediction_end_time = time.time()
                self.total_prediction_time += prediction_end_time - prediction_start_time
                print('prediction cost: ', prediction_end_time - prediction_start_time)

            elif(current_iteration >= 4 
                and current_iteration in self.prediction_checkpoint 
                and self.reliable_prediction[trial.trial_id]):

                prediction_start_time = time.time()

                mean, standard_deviation = self.predict_convergence_accuracy(current_iteration, trial.trial_id, current_accuracy)
                
                self.validate_mean(trial_runner, mean, trial.trial_id)
                self.update_prediction_history(trial.trial_id, mean, standard_deviation)

                # mean = random.uniform(0, 1)
                # standard_deviation = random.uniform(0, 1)
                if(self.exist(trial.trial_id, self.sorted_running_trials)):
                    self.update_sorted_running_trial(trial.trial_id, mean, standard_deviation)
                elif(self.exist(trial.trial_id, self.sorted_suspended_trials)):
                    self.update_sorted_suspended_trial(trial.trial_id, mean, standard_deviation)

                prediction_end_time = time.time()
                self.total_prediction_time += prediction_end_time - prediction_start_time
                print('prediction cost: ', prediction_end_time - prediction_start_time)


            self.schedule(trial_runner) 

            self.print_pbhs_info("OTR")

        print(trial.trial_id, "current iteration", current_iteration)
        print("duration since starting", time.time() - self.experiment_start_time)
        
        self.current_accuracy[trial.trial_id] = result[self.metric_key]
 
        self.experiment_current_time = time.time()
        if(self.experiment_time_limit - (self.experiment_current_time - self.experiment_start_time) < self.expected_trial_duration
                and not self.last_stage and not self.last_stage_launcher):
            print("launch last stage in ", trial.trial_id, " result")
            self.last_stage = True
            self.last_stage_launcher = True
            all_candidates = self.get_all_candidates()
            if(len(all_candidates) > 0):
                n_last_trials = min(len(all_candidates), int(self.total_gpus))
                self.last_trials_info = all_candidates[0 : n_last_trials]
            self.sorted_running_trials = []
            self.sorted_suspended_trials = []
            
        elif(self.last_stage and self.last_stage_launcher
                and self.exist(trial.trial_id, self.last_trials_info)):
            self.last_stage_launcher = False
            
        ###################### Phase2 #####################
        if(self.last_stage and self.last_stage_launcher and self.exist(trial.trial_id, self.last_trials_info)):
            print("~~~~~~~~~~~~~~Last stage 1~~~~~~~~~~~~~~", trial) 
            return TrialScheduler.CONTINUE
        elif(self.last_stage and not self.last_stage_launcher and self.exist(trial.trial_id, self.last_trials_info)):
            print("~~~~~~~~~~~~~~Last stage 2~~~~~~~~~~~~~~", trial)
            return TrialScheduler.CONTINUE
        elif(self.last_stage):
            return TrialScheduler.STOP
        ###################################################
        ###################### Phase1 #####################
        elif(self.exist(trial.trial_id, self.sorted_suspended_trials)):
            print("##Action: Pause trial ", trial.trial_id, "and release resource.")
            return TrialScheduler.PAUSE
        elif(self.exist(trial.trial_id, self.sorted_running_trials)):
            print("##Action: Continue trial ", trial.trial_id)
            return TrialScheduler.CONTINUE
        else:
            print("##Action: Stop trial ", trial.trial_id)
            if(trial.trial_id not in self.unpromising_trials):
                print("Report an exception")
            return TrialScheduler.STOP
        ###################################################

    def on_trial_complete(self, trial_runner: "trial_runner.TrialRunner",
                          trial: Trial, result: Dict):
        print("call on_trial_complete")
        self.completed_trials += 1
        
        # prediction_mean_error = [0, 0, 0]
        # if(not self.last_stage):
        #     print(self.completed_trials, " completed trials")
        #     for i in range(0, 3):
        #         self.total_prediction_error[i] += abs(result[self.metric_key] - self.prediction_history[trial.trial_id][i])
        #         prediction_mean_error[i] = self.total_prediction_error[i] / self.completed_trials
        #         print("Prediction mean error in ", self.prediction_checkpoint[i], "iteration: ", prediction_mean_error[i])

        if(self.last_stage and self.exist(trial.trial_id, self.last_trials_info)):
            print("total prediction cost:", self.total_prediction_time)
            print("total experiment time:", time.time() - self.experiment_start_time)
        if(self.exist(trial.trial_id, self.sorted_running_trials)):
            terminated_trial_index = self.find_trial_index(trial.trial_id, self.sorted_running_trials)
            self.sorted_running_trials.pop(terminated_trial_index)
        elif(self.exist(trial.trial_id, self.sorted_suspended_trials)):
            terminated_trial_index = self.find_trial_index(trial.trial_id, self.sorted_suspended_trials)
            self.sorted_suspended_trials.pop(terminated_trial_index)
        self.update_prediction_history(trial.trial_id, result[self.metric_key], 0.0)
        

    def on_trial_remove(self, trial_runner: "trial_runner.TrialRunner",
                        trial: Trial):
        print("call on_trial_remove")

    def choose_trial_to_run(
            self, trial_runner: "trial_runner.TrialRunner") -> Optional[Trial]:
        
        if(not self.last_stage):
            self.update_experiment_status()

        if(self.last_stage and self.last_stage_launcher):
            print('################# Start pbhs phase 2 ##################')
            trial_runner._search_alg._finished = True
            for trial in trial_runner.get_trials():
                if(not self.exist(trial.trial_id, self.last_trials_info) and (trial.status == Trial.PENDING or trial.status == Trial.PAUSED)):
                    trial.status = Trial.TERMINATED
                    return None

                if(not self.exist(trial.trial_id, self.last_trials_info) and trial.status != Trial.TERMINATED):
                    print("The last trials:", [x[0] for x in self.last_trials_info])
                    print("####Last stage#### wait for", trial.trial_id, trial.status, "to terminate")
                    return None

            self.last_stage_launcher = False 
            return None

        elif(self.last_stage and not self.last_stage_launcher):
            print("####Last stage##### ", self.last_trials_info, [trial_runner.get_trial(x[0]).status for x in self.last_trials_info])
            for trial in trial_runner.get_trials():
                if(not self.exist(trial.trial_id, self.last_trials_info) and (trial.status == Trial.PENDING or trial.status == Trial.PAUSED)):
                    print("####Last stage#### kill new/paused trial:", trial.status, trial.trial_id)
                    trial_runner.trial_executor.stop_trial(trial)

            for trial in trial_runner.get_trials():
                if(self.exist(trial.trial_id, self.last_trials_info) and (trial.status == Trial.PAUSED or trial.status == Trial.PENDING)):
                    return trial

            return None

        for trial_info in self.sorted_running_trials:
            trial = trial_runner.get_trial(trial_info[0])
            if (trial.status == Trial.PAUSED):
                # if(trial_runner.has_resources(trial.resources)): # this may report wrong resource information
                if(True):
                    print("$$restart trial: ", trial.trial_id)
                    return trial
                else:
                    print("No resources available for PAUSED trial", trial_info[0], "in running queue")
                    running_trials_counter = 0
                    for trial in trial_runner.get_trials():
                        if(trial.status == Trial.RUNNING):
                            running_trials_counter+=1
                    print("Total", running_trials_counter, "trials are running in the cluster")
                    running_trials_counter = 0
                    for item in self.sorted_suspended_trials:
                        if(trial_runner.get_trial(item[0]).status == Trial.RUNNING):
                            print(item[0], "in suspended queue is still running")
            elif (trial.status == Trial.PENDING):
                # if(trial_runner.has_resources(trial.resources)):
                if(True):
                    print("$$start trial: ", trial.trial_id)
                    return trial
                else:
                    print("No resources available for PENDING trial", trial_info[0], "in running queue")
                    running_trials_counter = 0
                    for trial in trial_runner.get_trials():
                        if(trial.status == Trial.RUNNING):
                            running_trials_counter+=1
                    print("Total", running_trials_counter, "trials are running in the cluster")
                    running_trials_counter = 0
                    for item in self.sorted_suspended_trials:
                        if(trial_runner.get_trial(item[0]).status == Trial.RUNNING):
                            print(item[0], "in suspended queue is still running")

        # for trial_info in self.sorted_running_trials:
        #     trial = trial_runner.get_trial(trial_info[0])
        #     if (trial.status == Trial.PENDING):
        #         print("$$start trial: ", trial.trial_id)
        #         return trial

        if (len(self.sorted_suspended_trials) > 0 
                and len(self.sorted_running_trials) < self.total_gpus and self.n_exploration_running_trials >= self.expected_explorative_gpus):
            trial_info = self.sorted_suspended_trials.pop(0)
            trial = trial_runner.get_trial(trial_info[0])
            if(trial.status == Trial.PAUSED):
                print("move", trial_info, "to running queue")
                self.update_sorted_running_trial(trial_info[0], trial_info[1], trial_info[2])
                # return trial

        for trial in trial_runner.get_trials():
            if (trial.status == Trial.PENDING 
                    and len(self.sorted_running_trials) < self.total_gpus):
                print("$$explore new trial: ", trial.trial_id)
                self.update_sorted_running_trial(trial.trial_id, 0.0, 0.0)
                # return trial

        running_trials_counter = 0
        for trial in trial_runner.get_trials():
            if(trial.status == Trial.RUNNING):
                running_trials_counter+=1
        print("Total", running_trials_counter, "trials are running in the cluster")
        return None

    def debug_string(self) -> str:
        print("Experiment time limit: " + str(self.experiment_time_limit) + 's')
        print("Exploration ratio: " + str(self.exploration_ratio * 100) + "%")
        return "Using PBHS scheduling algorithm."

    @property
    def expected_trial_duration(self):
        return (self.max_iteration) * np.median(self.single_iteration_times)

    @property
    def n_exploitation_running_trials(self):
        number_of_running_trials = len(self.sorted_running_trials)
        number_of_exploration_running_trials = 0
        
        for trial_info in self.sorted_running_trials:
            if(trial_info[1] == 0.0 and trial_info[2] == 0.0):
                number_of_exploration_running_trials += 1
        
        return number_of_running_trials - number_of_exploration_running_trials

    @property
    def n_exploration_running_trials(self):
        return len(self.sorted_running_trials) - self.n_exploitation_running_trials
    
    @property
    def expected_explorative_gpus(self):
        return int((self.total_gpus + 1) * self.exploration_ratio)
    
    @property
    def expected_exploitative_gpus(self):
        return self.total_gpus - self.expected_explorative_gpus



if __name__ == "__main__":
    scheduler = PBHS()

    