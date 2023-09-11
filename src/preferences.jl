mutable struct PreferenceDist
    name_param_map::Dict # map parameters to the right corresponding variables
    dims::Int # helps to know how big the matrix should be 
    misc
end


function draw_normal!(array_to_fill, df, preferences::PreferenceDist; N = 100)
    names = preference.name_param_map;
    # All params should be stored as either sigma_X or sigma_X_Y
    names = string.(names);
    for i ∈ eachindex(names)

    end
    
    array_to_fill 
end

# TO-DO
# function draw_lognormal!(array_to_fill, df, preferences::PreferenceDist; N = 100, FRAC = true)
#     # if FRAC, then have to first map variances and covariances to parameters of the lognormal distribution

#     # Else, they should represent the Sigma from the actual multi-variate lognormal
# end