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

    greedy_rewards = Array{Float64}(undef,steps,runs)
    eps1_rewards = Array{Float64}(undef,steps,runs)
    eps2_rewards = Array{Float64}(undef,steps,runs)
    # cumulative_rewards = Array{Float64}(undef,steps)
    # cumulative_average = Array{Float64}(undef,steps)

    eps1 = .1
    eps2 = .01

    # Main loop
    for jj=1:runs
        # Q array of reward estimates
        Q_greedy = [0.0 for i in 1:k]
        Q_eps1 = [0.0 for i in 1:k]
        Q_eps2 = [0.0 for i in 1:k]

        # Counts how many times Q has been called (n):
        Qn_greedy = [0 for i in 1:k]
        Qn_eps1 = [0 for i in 1:k]
        Qn_eps2 = [0 for i in 1:k]

        # Simulate rewards
        # ith row is ith time step, jth column is jth run

        for ii=1:steps

            # Choose action
            action_greedy = totally_greedy(Q_greedy,Qn_greedy)
            action_eps1 = explore(Q_eps1,Qn_eps1,eps1)
            action_eps2 = explore(Q_eps2,Qn_eps2,eps2)

            # Store rewards
            greedy_rewards[ii,jj] = rand(distributions[action_greedy])
            eps1_rewards[ii,jj] = rand(distributions[action_eps1])
            eps2_rewards[ii,jj] = rand(distributions[action_eps2])

            # println(rewards[ii,jj])
            
            # Update estimate of q*
            if Qn_greedy[action_greedy] > 0
                # Normal update
                Q_greedy[action_greedy] += 1/Qn_greedy[action_greedy]*(greedy_rewards[ii,jj] - Q_greedy[action_greedy])
                Qn_greedy[action_greedy] += 1
            else
                # Just increment Qn_eps1
                Q_greedy[action_greedy] = greedy_rewards[ii,jj]
                Qn_greedy[action_greedy] += 1
            end

            if Qn_eps1[action_eps1] > 0
                # Normal update
                Q_eps1[action_eps1] += 1/Qn_eps1[action_eps1]*(eps1_rewards[ii,jj] - Q_eps1[action_eps1])
                Qn_eps1[action_eps1] += 1
            else
                # Just increment Qn
                Q_eps1[action_eps1] = eps1_rewards[ii,jj]
                Qn_eps1[action_eps1] += 1
            end

            if Qn_eps2[action_eps2] > 0
                # Normal update
                Q_eps2[action_eps2] += 1/Qn_eps2[action_eps2]*(eps2_rewards[ii,jj] - Q_eps2[action_eps2])
                Qn_eps2[action_eps2] += 1
            else
                # Just increment Qn
                Q_eps2[action_eps2] = eps2_rewards[ii,jj]
                Qn_eps2[action_eps2] += 1
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
    avg_rewards_greedy = Array{Float64}(undef,steps)
    avg_rewards_eps1 = Array{Float64}(undef,steps)
    avg_rewards_eps2 = Array{Float64}(undef,steps)

    for kk=1:steps
        avg_rewards_greedy[kk] = mean(greedy_rewards[kk,:])
        avg_rewards_eps1[kk] = mean(eps1_rewards[kk,:])
        avg_rewards_eps2[kk] = mean(eps2_rewards[kk,:])
    end

    plot(1:(steps+1), vcat(0.0,avg_rewards_greedy), linewidth=1, label="Greedy");
    plot(1:(steps+1), vcat(0.0,avg_rewards_eps1), linewidth=1, label=string("Eps1 = ", eps1));
    plot(1:(steps+1), vcat(0.0,avg_rewards_eps2), linewidth=1, label=string("Eps2 = ", eps2));

    # Max reward
    plot(1:(steps+1), [maximum(means) for i in 1:(steps+1)], linewidth=1, linestyle="--", label="Maximum")
    # display(sum_plot)

    legend()
    means

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
        # println("explore_bool == 1")
        return rand(1:length(Q))
    else
        # Greedy action
        # println("greedy action")
        # println("START")
        # println(Q)
        max_vec = findall(Q .== maximum(Q))
        # println(max_vec)
        # println(max_vec[rand(1:length(max_vec))])
        # println("END")
        return max_vec[rand(1:length(max_vec))]
    end
end