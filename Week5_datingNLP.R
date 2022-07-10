# yossi shrem 308492503
# Ex_5

library(stringr)
library(dplyr)
library(ggplot2)
library(mosaic)
library(stringr)
library(xtable)
library(gridExtra)
library(stopwords)
library(quanteda)
library(parallel)
library(caret)
library(rpart)
library(Matrix)

# Set rounding to 2 digits
options(digits = 2)

## 1. Load texts

# Load applicants' profiles
profiles <- read.csv(file.path("dating", "okcupid_profiles.csv"),
                     header = TRUE,
                     stringsAsFactors = FALSE,
)
dim(profiles)

# Depict number of male and female
table(profiles$sex)

# Select the essays columns
essays <- select(profiles, starts_with("essay"))

# Combine all essays of user into one essay
essays <- apply(essays, MARGIN = 1, FUN = paste, collapse = " ")

# Add 'essays' column to the Data Frame containing the full essay for each user
profiles$essays <- essays

# 2. Clean text, tokenize, remove stopwords 

# Tokenize essay texts
all.tokens <- tokens(essays,
                     what = "word",
                     remove_numbers = TRUE, remove_punct = TRUE,
                     remove_symbols = TRUE, remove_hyphens = TRUE
)

# Lower case the tokens
all.tokens <- tokens_tolower(all.tokens)

# list languages for a specific source
stopwords::stopwords_getlanguages("snowball")


head(stopwords::stopwords("english"), 20)

#Remove stopwords
all.tokens <- tokens_select(all.tokens, stopwords(),
                            selection = "remove"
)

# 3. Perform stemming on the tokens.
all.tokens <- tokens_wordstem(all.tokens, language = "english")

# 4. DFM - document-term frequency matrix
all.tokens.dfm <- dfm(all.tokens, tolower = TRUE)

all.tokens.dfm[1:5, 1:5]
dim(all.tokens.dfm)

# Trim
all.tokens.dfm <- dfm_trim(all.tokens.dfm, min_termfreq = 1000, min_docfreq = 10000)
dim(all.tokens.dfm)


# 4.TF- IDF

all.tokens.matrix <- Matrix(all.tokens.dfm, sparse = TRUE)


# TF function
term.frequency <- function(row) {
  row / sum(row)
}

# IDF function
inverse.doc.freq <- function(column) {
  N <- length(column)
  doc.count <- length(which(column > 0))
  
  results <- log10(N / doc.count)
  results[is.infinite(results)] <- -99999
  results
}

# TF- IDF function
tfidf <- function(tf, idf) {
  tf * idf
}

# TF matrix
all.tokens.tf <- apply(all.tokens.matrix, MARGIN = 1, FUN = term.frequency)
all.tokens.tf[1:5, 1:5]

# IDF vector
all.tokens.idf <- apply(all.tokens.matrix, MARGIN = 2, FUN = inverse.doc.freq)
all.tokens.idf

gc()

# TF- IDF matrix
all.tokens.tfidf <- t(apply(all.tokens.tf, MARGIN = 2, FUN = tfidf, idf = all.tokens.idf))
all.tokens.tfidf[1:5, 1:5]

# Convert na to 0
all.tokens.tfidf[is.na(all.tokens.tfidf)] <- 0


# 5. Train a model to identify male vs. female by their text

# Convert to dataframe
all.tokens.tfidf.df <- as.data.frame(all.tokens.tfidf)

# Rectify the names of variables
names(all.tokens.tfidf.df) <- make.names(names(all.tokens.tfidf.df))

# Add labels of male\ female to the DFM data frame
all.tokens.tfidf.df$sex <- profiles$sex

# 10 folds cross validation 3 times- repeatedcv
set.seed(100)

# Training parameters
control <- trainControl(method = "repeatedcv", number = 10, repeats = 3, savePredictions = TRUE)

start_time <- Sys.time()

# Training the model
model <- train(
  x = all.tokens.tfidf.df[ , 1:ncol(all.tokens.tfidf.df)-1],
  y = all.tokens.tfidf.df[ , ncol(all.tokens.tfidf.df)],
  method = "rpart",
  trControl = control
)

end_time <- Sys.time()
end_time - start_time

# Test the model by calculating confusion matrix
tree.model.confusionMatrix <- confusionMatrix(model)
# Since entries are percentual average cell counts across resamples- multiply by length
tree.model.confusionMatrix$table <- tree.model.confusionMatrix$table * length(all.tokens.tfidf.df)
tree.model.confusionMatrix

#6. Remove the batch effect: find and eliminate "male\ female words"
batch.words <- names(model$finalModel$variable.importance)
batch.words

tokens.set <- all.tokens.tfidf.df[, !(names(all.tokens.tfidf.df) %in% batch.words)]
tokens.set[1:5, 1:5]

# 7. Cluster the applicants to 2,3,4 and 10 clusters

pca <- prcomp(tokens.set[ , 1:ncol(tokens.set)-1], scale = TRUE)

# Color palette
rbPal <- colorRampPalette(c("red", "blue"))

pdf("Week5_datingNLP.pdf")
for (k in c(2, 3, 4, 10)) {
  # Clustering
  clusters <- kmeans(tokens.set[ , 1:ncol(tokens.set)-1], k, iter.max = 100)
  print(table(clusters$cluster))
  plot(pca$x[, 1], pca$x[, 2], col = rbPal(k)[clusters$cluster])
  title(paste("PCA colored by clusters k=", k))
}
dev.off()

final.model <- model$finalModel
save(
  final.model,
  all.tokens.tfidf.df,
  pca,
  file = "Week5_datingNLP.rdata"
)