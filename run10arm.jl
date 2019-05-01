#= 
Multi-armed Bandit problem

Simulates the k-armed bandit problem
=#

using Distributions
using PyPlot
using Statistics


function karm_bandit(k::Int64=10, steps::Int64=1000, runs::Int64=1)
# k : Number of arms
    # mean_range : range of values that the mean for each arm is sampled from
    # steps : number of steps the simulation runs for.

    # Array of distributions
    means = [m for m in rand(Normal(),k)]
    distributions = [Normal(m) for m in means]

    # Q array of reward estimates
    Q = [0.0 for i in 1:k]

    # Counts how many times Q has been called (n):
    Qn = [0 for i in 1:k]

    # Simulate rewards
    # ith row is ith time step, jth column is jth run
    greedy_rewards = Array{Float64}(undef,steps,runs)
    eps1_rewards = Array{Float64}(undef,steps,runs)
    eps2_rewards = Array{Float64}(undef,steps,runs)
    # cumulative_rewards = Array{Float64}(undef,steps)
    # cumulative_average = Array{Float64}(undef,steps)

    eps1 = .1
    eps2 = .05

    # Main loop
    for jj=1:runs
        for ii=1:steps

            # Choose action
            action1 = totally_greedy(Q,Qn)

            # Store rewards
            rewards[ii,jj] = rand(distributions[action])
            # println(rewards[ii,jj])
            
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

    plot(1:(steps+1), vcat(0.0,avg_rewards), linewidth=2);

    # Max reward
    plot(1:(steps+1), [maximum(means) for i in 1:(steps+1)], linewidth=1, linestyle="--")
    # display(sum_plot)

    return avg_rewards, means, Q, Qn

end


function totally_greedy(Q,Qn)::Int64
    # Make sure all options are chosen at least once
    # println(length(findall(Qn .== 0)))
    # if length(findall(Qn .== 0)) > 0
    #     unexplored_vec = findall(Qn .== 0)
    #     # println("length(findall(Qn .== 0)) < 0")
    #     return unexplored_vec[rand(1:length(unexplored_vec))]
    # else
        if length(findall(Q .== maximum(Q))) == 1
            # println("length(findall(Q .== maximum(Q))) == 1")
            return argmax(Q)
        else
            # Choose randomly among largest values
            # println("else")
            max_vec = findall(Q .== maximum(Q))
            return max_vec[rand(1:length(max_vec))]
        end
    # end
end


function explore(Q,Qn,eps)::Int64
    explore_bool = rand(Binomial(1,eps)) # Takes exploration action with probability 1 - eps. Binomial
    if explore_bool == 1
        # Exploration action
        return rand(1:length(Q))
    else
        # Greedy action
        max_vec = findall(Q .== maximum(Q))
        return max_vec[rand(1:length(max_vec))]
    end
end

