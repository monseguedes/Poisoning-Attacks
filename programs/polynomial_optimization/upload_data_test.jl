using CSV
using DataFrames

config = Dict("no_training_samples" => 100,
              "poison_rate" => 4,
              "no_numerical_features" => 2,
              "no_categorical_features" => 2,
              "poison_start" => 5,
              "regularization" => 0.1)

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


df2 = get_training_dataframe(data, config)
println(df2)

poison_df = get_poison_dataframe(df2, config)
println(poison_df)

print(get_training_target(df2))

# matrix = get_training_numerical_array(df2, config)
# print(size(matrix))

# matrix = get_training_categorical_array(df2, config)
# print(size(matrix))

categories_dict = get_no_categories_dict(data, config)
print(categories_dict)  



training_dataframe = get_training_dataframe(data, config)
poison_dataframe = get_poison_dataframe(training_dataframe, config)

# Parameters--------------------------------------------------
# Training data
num_training = get_numerical_array(training_dataframe, config)
cat_training = get_categorical_array(training_dataframe, config)
y_training = get_target_array(training_dataframe)

# Poisoning data
cat_poison = get_categorical_array(poison_dataframe, config)
y_poison = get_target_array(poison_dataframe)

println(size(num_training))

println(size(cat_training))

println(size(cat_poison))

no_poison_samples = size(poison_dataframe, 1)
println(no_poison_samples)





