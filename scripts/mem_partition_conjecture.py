from grl.utils.policy_eval import lstdq_lambda



if __name__ == "__main__":
    # TODO: load dataset, pomdp
    # TODO: calculate returns over dataset

    lambda_v_vals, lambda_q_vals, _ = lstdq_lambda(pi, pomdp, lambda_=lambda_)

    eps_mem_lambda_v_vals, eps_mem_lambda_q_vals, _ = lstdq_lambda(mem_aug_pi, eps_mem_aug_pomdp, lambda_=lambda_)
