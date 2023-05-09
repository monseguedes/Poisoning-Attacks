"""
Code to build the continuous version of 
poisoning attacks.
"""

using DynamicPolynomials
using SumOfSquares
using Ipopt
using Gurobi
using Random
using LinearAlgebra
using CSV

config = Dict("no_training_samples" => 5,
              "poison_rate" => 20,
              "no_numerical_features" => 1,
              "no_categorical_features" => 1,
              "poison_start" => 1,
              "regularization" => 0.1)

### Data-------------------------------------------------------------------------------------
# Read a CSV file
data = CSV.read("../../data/fake/data-binary.csv", DataFrame)

function get_training_dataframe(df, config)
   # Select columns
   numerical_columns = [string(i) for i in 1:config["no_numerical_features"]]
   prefix = [string(i) * ":" for i in 1:config["no_categorical_features"]]
   categorical_columns = filter(col -> any([startswith(col, prefix[i]) for i in 1:length(prefix)]), names(df))
   df_selected = select(df, vcat(numerical_columns, categorical_columns, ["target"]))
   # Select rows
   df_subset = df_selected[1:config["no_training_samples"], :]

   return df_subset
end

function get_no_categories_dict(df, config)
   prefix = [string(i) * ":" for i in 1:config["no_categorical_features"]]
   categorical_columns = filter(col -> any([startswith(col, prefix[i]) for i in 1:length(prefix)]), names(df))
   no_categories_dict = Dict()
   for col in categorical_columns
       prefix = split(col, ":")[1]
       if haskey(no_categories_dict, prefix)
           no_categories_dict[prefix] += 1
       else
           no_categories_dict[prefix] = 1
       end
   end
   return no_categories_dict
end
   
function get_poison_dataframe(df, config)
   # Select rows 
   no_poison_samples = Int(config["no_training_samples"] * (config["poison_rate"] / 100))
   df_poison = df[config["poison_start"]:config["poison_start"] + no_poison_samples, :]
   return df_poison
end

function get_target_array(df)
   return df[:, "target"]
end

function get_numerical_array(df, config)
   numerical_columns = [string(i) for i in 1:config["no_numerical_features"]]
   return Matrix(values(df[:, numerical_columns]))
end

function get_categorical_array(df, config)
   prefix = [string(i) * ":" for i in 1:config["no_categorical_features"]]
   categorical_columns = filter(col -> any([startswith(col, prefix[i]) for i in 1:length(prefix)]), names(df))
   return Matrix(values(df[:, categorical_columns]))
end

training_dataframe = get_training_dataframe(data, config)
poison_dataframe = get_poison_dataframe(training_dataframe, config)

### NLP model-------------------------------------------------------------------------------

# Parameters--------------------------------------------------
# Training data
num_training = get_numerical_array(training_dataframe, config)
cat_training = get_categorical_array(training_dataframe, config)
y_training = get_target_array(training_dataframe)

# Poisoning data
cat_poison = get_categorical_array(poison_dataframe, config)
y_poison = get_target_array(poison_dataframe)

no_numerical_features = config["no_numerical_features"]
no_categorical_columns = size(cat_training, 2)
no_training_samples = config["no_training_samples"]
no_poison_samples = size(poison_dataframe, 1)

# Model
model = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))
# Variables
@variable(model, 0 <= num_poison[1:no_poison_samples, 1:no_numerical_features] <= 1)
@variable(model, num_weights[1:no_numerical_features])
@variable(model, cat_weights[1:no_categorical_columns])
@variable(model, bias)
@variable(model, t[1:no_poison_samples])

@objective(model, Min, - sum((dot(num_weights, num_training[j, :]) 
                            + dot(cat_weights, cat_training[j, :]) + bias - y_training[j])^2 for j in 1:no_training_samples))
                      

# Constraints
# Derivative of lower-level with respect to numerical weights
for j = 1:no_numerical_features
    @NLconstraint(model, 2/3 * (sum((sum(num_weights[i] * num_training[n, i] for i in 1:no_numerical_features) 
                                   + sum(cat_weights[i] * cat_training[n, i] for i in 1:no_categorical_columns) + bias - y_training[n]) * num_training[n,j] for n in 1:no_training_samples)
                                   + sum(t[i] * num_poison[i, j] for i in 1:no_poison_samples)) + 2 * lambda * num_weights[j] == 0)
end
# Derivative of lower-level with respect to categorical weights
for j = 1:no_categorical_columns
    @NLconstraint(model, 2/3 * (sum((sum(num_weights[i] * num_training[n, i] for i in 1:no_numerical_features) 
                                   + sum(cat_weights[i] * cat_training[n, i] for i in 1:no_categorical_columns) + bias - y_training[n]) * cat_training[n,j] for n in 1:no_training_samples)
                                   + sum(t[i] * cat_poison[i, j] for i in 1:no_poison_samples)) + 2 * lambda * cat_weights[j] == 0)
end
# Derivative of lower-level with respect to bias
@NLconstraint(model, 2/3 * (sum(sum(num_weights[i] * num_training[n, i] for i in 1:no_numerical_features) 
                              + sum(cat_weights[i] * cat_training[n, i] for i in 1:no_categorical_columns) + bias - y_training[n] for n in 1:no_training_samples)
                              + sum(t[i] for i in no_poison_samples)) == 0)
# Substitution of predictions to allow first level of hierarchy
for j = 1:no_poison_samples
   @NLconstraint(model, t[j] - (sum(num_weights[i] * num_poison[j, i] for i in 1:no_numerical_features) 
                              + sum(cat_weights[i] * cat_poison[j, i] for i in 1:no_categorical_columns) + bias - y_poison[j]) == 0)  
end

optimize!(model)
solution_summary(model)
@show objective_value(model)



### Polynomial model---------------------------------------------------------------------------
no_variables = no_categorical_columns + no_numerical_features + 1 + (no_categorical_columns + no_numerical_features) * no_poison_samples + no_poison_samples
println("Number is variables is $(no_variables)")

@polyvar num_weights[1:no_numerical_features] cat_weights[1:no_categorical_columns] num_poison[1:no_poison_samples, 1:no_numerical_features] bias t[1:no_poison_samples] cat_poison[1:no_poison_samples, 1:no_categorical_columns]

p = -(sum((sum(num_weights[i] * num_training[j, i] for i in 1:no_numerical_features) 
         + sum(cat_weights[i] * cat_training[j, i] for i in 1:no_categorical_columns) + bias - y_training[j])^2 for j in 1:no_training_samples))

S = @set num_poison[1, 1]*(1-num_poison[1, 1]) >= 0  

for j in 1:no_numerical_features    
   for i in 2:no_poison_samples
      global S = S ∩ @set(num_poison[i, j] * (1-num_poison[i, j]) >= 0)
   end
end      

# TODO add SOS1 constraints
# sum(cat_poison[i] for i in 1:2) == 1 &&
# sum(cat_poison[i] for i in 2:4) == 1 

for j in 1:no_categorical_columns
   for i in 2:no_poison_samples   
      global S = S ∩ @set(cat_poison[i, j] * (cat_poison[i, j] - 1) == 0)
   end
end
for j in 1:no_numerical_features    # Num weights constraints
   global S = S ∩ @set( 2/3 * (sum((sum(num_weights[i] * num_training[n, i] for i in 1:no_numerical_features) 
                                  + sum(cat_weights[i] * cat_training[n, i] for i in 1:no_categorical_columns) + bias - y_training[n]) * num_training[n,j] for n in 1:no_training_samples)
                                  + sum(t[i] * num_poison[i, j] for i in 1:no_poison_samples)) + 2 * lambda * num_weights[j] == 0)
end
for j in 1:no_categorical_columns    # Cat weights constraints
   global S = S ∩ @set(2/3 * (sum((sum(num_weights[i] * num_training[n, i] for i in 1:no_numerical_features) 
                                 + sum(cat_weights[i] * cat_training[n, i] for i in 1:no_categorical_columns) + bias - y_training[n]) * cat_training[n,j] for n in 1:no_training_samples)
                                 + sum(t[i] * cat_poison[i, j] for i in 1:no_poison_samples)) + 2 * lambda * cat_weights[j] == 0)
end
# Bias constraint
S = S ∩ @set(2/3 * (sum(sum(num_weights[i] * num_training[n, i] for i in 1:no_numerical_features) 
                      + sum(cat_weights[i] * cat_training[n, i] for i in 1:no_categorical_columns) + bias - y_training[n] for n in 1:no_training_samples)
                      + sum(t[i] for i in 1:no_poison_samples)) == 0)
# t constraint
for j = 1:no_poison_samples
   global S = S ∩ @set(t[j] - (sum(num_weights[i] * num_poison[j, i] for i in 1:no_numerical_features) 
                          + sum(cat_weights[i] * cat_poison[j, i] for i in 1:no_categorical_columns) + bias - y_poison[j]) == 0)  
end

using MosekTools

solver = optimizer_with_attributes(Mosek.Optimizer, MOI.Silent() => false)

model = SOSModel(solver)
@variable(model, α)
@objective(model, Max, α)
@constraint(model, p >= α, domain = S)
optimize!(model)
@show termination_status(model)
@show objective_value(model)
