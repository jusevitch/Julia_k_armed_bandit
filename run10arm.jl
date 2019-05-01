#= 
Multi-armed Bandit problem

Simulates the k-armed bandit problem
=#

using Distributions
using PyPlot
using Statistics


function karm_bandit(k::Int64=10, mean_range=10, steps::Int64=1000, runs::Int64=1)
# k : Number of arms
    # mean_range : range of values that the mean for each arm is sampled from
    # steps : number of steps the simulation runs for.

    # Array of distributions
    distributions = [Normal(m) for m in (mean_range*rand(k) .- mean_range / 2)]


    # Q array of reward estimates
    Q = [0.0 for i in 1:k]

    # Counts how many times Q has been called (n):
    Qn = [0 for i in 1:k]

    # Simulate rewards
    # ith row is ith time step, jth column is jth run
    rewards = Array{Float64}(undef,steps,runs)
    # cumulative_rewards = Array{Float64}(undef,steps)
    # cumulative_average = Array{Float64}(undef,steps)
    for jj=1:runs
        for ii=1:steps

            # Choose action
            action = totally_greedy(Q,Qn)

            # Store rewards
            rewards[ii,jj] = rand(distributions[action])
            
            # Update estimate of q*
            if Qn[action] > 0
                # Normal update
                Q[action] += 1/Qn[action]*(rewards[ii,jj] - Q[action])
                Qn[action] += 1
            else
                # Just increment Qn
                Qn[action] += 1
            end
            
            # if ii == 1
            #     cumulative_rewards[ii] = rewards[ii]
            #     cumulative_average[ii] = rewards[ii]
            # else
            #     cumulative_rewards[ii] = cumulative_rewards[ii-1] + rewards[ii]
            #     cumulative_average[ii] = cumulative_rewards[ii] / ii
            # end
        end
    end

    # Create average reward over all runs
    avg_rewards = Array{Float64}(undef,steps)
    for kk=1:steps
        avg_rewards[kk] = mean(rewards[kk,:])
    end

    sum_plot = plot(1:steps, avg_rewards, linewidth=2);
    display(sum_plot)

    return rewards

end


function totally_greedy(Q,Qn)::Int64
    # Make sure all options are chosen at least once
    if length(findall(Qn .== 0)) < 0
        unexplored_vec = findall(Qn .== 0)
        return unexplored_vec[rand(1:length(unexplored_vec))]
    else
        if length(findall(Q .== maximum(Q))) == 1
            return argmax(Q)
        else
            # Choose randomly among largest values
            max_vec = findall(Q .== maximum(Q))
            return max_vec[rand(1:length(max_vec))]
        end
    end
end


function my_test_function()
    return 1
end