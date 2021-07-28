
f_logLikelihood <- function(M,x_train,uniform_dist,u,sigma)
{
  logLike <- sum(log(M[[1]]*uniform_dist + M[[2]]*dnorm(x_train, u, sqrt(sigma))))
  return (logLike)
}
  
f_updateResposibilities <- function(M,x_train,uniform_dist,u,sigma)
{
  uniformP <- uniform_dist
  gaussP <- dnorm(x_train, u, sqrt(sigma)) #stats.norm.pdf(x_train,u,sqrt(sigma))
  resposibilities <- cbind( (M[[1]]*uniformP / (M[[1]]*uniformP + M[[2]]*gaussP)), #.reshape(-1,1),
                                (M[[2]]*gaussP / (M[[1]]*uniformP + M[[2]]*gaussP)) ) # .reshape(-1,1) ))
  return (resposibilities)
}
  
f_updateMixtureParams <- function(resposibilities,M,x_train)
{
  M[[1]] <- sum(resposibilities[,1])/sum(resposibilities)
  M[[2]] <- sum(resposibilities[,2])/sum(resposibilities)
  u <- (sum(resposibilities[,2]*x_train))/sum(resposibilities[,2])
  sigma <- sum((x_train-u)**2*resposibilities[,2])/sum(resposibilities[,2])
  return (list("M" = M,
               #"uniform_dist" = uniform_dist,
               "u" = u,
               "sigma" = sigma))
} 

f_calcCutoffs <- function(M,x_train,u,sigma)
{
  tmp <- sqrt(-2*sqrt(sigma)**2 * (log(M[[1]]/M[[2]]) - log(max(x_train)-min(x_train)) + log(sqrt(sigma)) + 0.5*log(2*pi)))
  cutoffs <- c("low" = u-tmp,
               "high" = u+tmp)
  return (cutoffs)
}
  
f_call_changes <- function(all_t_tests)
{
  # ignore DEAD and GHOST columns
  localization_cols <- colnames(all_t_tests)[ !colnames(all_t_tests) %in% c("GHOST", "DEAD")]
  tscores <- all_t_tests[ ,localization_cols ]

  localization_hits = list()
  Outlier_M <- 0.001
  for (loc in localization_cols)
  {
    x_train = tscores[,loc]
    ### remove BLANK
    x_train <- x_train[ grep("BLANK", names(x_train), invert=T)]
    ### end remove BLANK
    M <- c(Outlier_M, 1-Outlier_M)
    #randInd = np.random.rand(len(x_train))
    resposibilities <- cbind(0.5 * rep(1, length(x_train)), 
                             0.5 * rep(1, length(x_train)))
    uniform_dist <- rep(1, length(x_train))/(max(x_train) - min(x_train))
    u <- 0
    sigma <- 0.1**2
    loglike_prev <- 1
    loglike <- 10
    i <- 0
    while (abs(loglike_prev-loglike) > 0.1)
    {
      loglike_prev <- loglike
      resposibilities <- f_updateResposibilities(M, x_train, uniform_dist, u, sigma)
      res_f_upd <- f_updateMixtureParams(resposibilities, M, x_train)
      u <- res_f_upd$u
      sigma <- res_f_upd$sigma
      #M = [Outlier_M, 1.-Outlier_M] # I do not update this parameter, therefore we do not need to reset it
      loglike <- f_logLikelihood(M, x_train, uniform_dist, u, sigma)
      i <- i+1
    }
    cutoffs <- f_calcCutoffs(M,x_train,u,sigma)
    localization_hits[[loc]] <- all_t_tests[ (all_t_tests[,loc] < cutoffs[["low"]]) |
                                          (all_t_tests[,loc] > cutoffs[["high"]]), ]
  }
  sapply(localization_hits, nrow)
  sum(sapply(localization_hits, nrow))
  sum(sapply(localization_hits, function(x) { length(grep("BLANK", row.names(x), invert=T)) }))
  df <- do.call("rbind", localization_hits)
  nrow(df)
  nrow(unique(df))
  nrow(unique(df[ grep("BLANK", row.names(df), invert = T),]))
  return (localication_hits)
}

f_get_significant_changes <- function(t_tests, Outlier_M = 0.001, VERBOSE=F)
{
  compartments <- colnames(t_tests)
  localization_hits = list()
  significant_changes <- list()
  significant_mut_higher <- list()
  significant_WT_higher <- list()
  cutoffs <- list()
  for (loc in compartments)
  {
    x_train = t_tests[,loc]
    M <- c(Outlier_M, 1-Outlier_M)
    resposibilities <- cbind(0.5 * rep(1, length(x_train)), 
                             0.5 * rep(1, length(x_train)))
    uniform_dist <- rep(1, length(x_train))/(max(x_train) - min(x_train))
    u <- 0
    sigma <- 0.1**2
    loglike_prev <- 1
    loglike <- 10
    i <- 0
    while (abs(loglike_prev-loglike) > 0.1)
    {
      loglike_prev <- loglike
      resposibilities <- f_updateResposibilities(M, x_train, uniform_dist, u, sigma)
      res_f_upd <- f_updateMixtureParams(resposibilities, M, x_train)
      u <- res_f_upd$u
      sigma <- res_f_upd$sigma
      #M = [Outlier_M, 1.-Outlier_M] # I do not update this parameter, therefore we do not need to reset it
      loglike <- f_logLikelihood(M, x_train, uniform_dist, u, sigma)
      i <- i+1
    }
    cutoffs[[loc]] <- f_calcCutoffs(M,x_train,u,sigma)
    cutoffs[[loc]][["u"]] <- u
    cutoffs[[loc]][["sigma"]] <- sigma
    localization_hits[[loc]] <- t_tests[ (t_tests[,loc] < cutoffs[[loc]][["low"]]) |
                                         (t_tests[,loc] > cutoffs[[loc]][["high"]]), ,
                                         drop=FALSE]
    cutoffs[[loc]][["counts_changes"]] <- nrow(localization_hits[[loc]])
    cutoffs[[loc]][["counts_mut_higher"]] <- sum(t_tests[,loc] < cutoffs[[loc]][["low"]])
    cutoffs[[loc]][["counts_WT_higher"]] <- sum(t_tests[,loc] > cutoffs[[loc]][["high"]])
    significant_changes[[loc]] <- sort(row.names(localization_hits[[loc]]))
    significant_mut_higher[[loc]] <- sort(row.names(t_tests)[ t_tests[,loc] < cutoffs[[loc]][["low"]] ])
    significant_WT_higher[[loc]] <- sort(row.names(t_tests)[ t_tests[,loc] > cutoffs[[loc]][["high"]] ])
    if (VERBOSE) { cat(sprintf("%s\t%s\t%s\t%s\t%s\t%s\n", loc, u, sigma, cutoffs[[loc]][["low"]], cutoffs[[loc]][["high"]], sum((t_tests[,loc] < cutoffs[["low"]]) | (t_tests[,loc] > cutoffs[["high"]]) ) ) ) }
  }
  # table to return w/ all t-tests, and adding the significant change in each row (we only show rows w/ at least a significant change)
  df <- unique(as.data.frame(do.call("rbind", localization_hits),stringsAsFactors = F))
  tmp <- do.call("rbind", sapply(names(significant_changes), function(x) { cbind(x, significant_changes[[x]] )  }) )
  locs_per_ORF <- sapply(split(tmp[,1], tmp[,2]), function(x) { paste(x, collapse=",") })
  df$enriched_loc <- locs_per_ORF[ row.names(df) ]
  # summary
  df_summary <- as.data.frame(do.call("rbind", cutoffs), stringsAsFactors = F)
  df_summary$genes_changing <- sapply(significant_changes[ row.names(df_summary) ], function(x) { paste(x, collapse=",") })
  df_summary$genes_mut_higher <- sapply(significant_mut_higher[ row.names(df_summary) ], function(x) { paste(x, collapse=",") })
  df_summary$genes_WT_higher <- sapply(significant_WT_higher[ row.names(df_summary) ], function(x) { paste(x, collapse=",") })
  return (list("df" = df,
               "df_summary" = df_summary))
}
