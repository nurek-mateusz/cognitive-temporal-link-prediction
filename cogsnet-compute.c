// Notes:
// #1 when compiling using GCC, "-lm" compilation option is required
// #2 all the events have to be in chronological order, otherwise CogSNet calculations will be wrong
// #3 the pathEvents must contain the header and an empty line at the end
// #4 output file with CogSNet is not sorted by weights or anything else

#define _XOPEN_SOURCE
#define LEN(arr) ((int) (sizeof (arr) / sizeof (arr)[0]))

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <errno.h>

// linear forgetting
float compute_weight_linear(int new_event, float weight_last_event, float time_difference, float lambda, float mu) {
	if(new_event == 1) {
		return(mu + (weight_last_event - time_difference * lambda) * (1 - mu));
	} else {
		return(weight_last_event - time_difference * lambda);
	}
}

// power forgetting
float compute_weight_power(int new_event, float weight_last_event, float time_difference, float lambda, float mu) {
	// We need to check whether the time_difference is greater or equal one,
	// since the power to sth smaller than one will result with heigher weight

	if(time_difference >= 1) {
		if(new_event == 1) {
			return(mu + (weight_last_event * pow(time_difference, -1 * lambda) * (1 - mu)));
		} else {
			return(weight_last_event * pow(time_difference, -1 * lambda));
		}
	} else {
			return(weight_last_event);
	}
}

// exponential forgetting
float compute_weight_exponential(int new_event, float weight_last_event, float time_difference, float lambda, float mu) {
	if(new_event == 1) {
		return(mu + (weight_last_event * exp(-1 * lambda * time_difference) * (1 - mu)));
	} else {
		return(weight_last_event * exp(-1 * lambda * time_difference));
	}
}

// computing the weight, invoked for every new event and at the end (surveys' dates)
float compute_weight(int time_to_compute, int time_last_event, char forgettingType[16], float weight_last_event, int new_event, float mu, float lambda, float theta, int units) {
	// Compute the time difference between events
	float time_difference = ((float)(time_to_compute - time_last_event) / (float)units);

	if (time_difference >= 0) {
		// First, we need to know how much time passed since last event
		// The time difference has to be zero or a positive value

		// Our new weigth of the edge
		float weight_new;

		if(strncmp(forgettingType, "linear", 6) == 0) {
			weight_new = compute_weight_linear(new_event, weight_last_event, time_difference, lambda, mu);
		} else if(strncmp(forgettingType, "power", 5) == 0) {
			weight_new = compute_weight_power(new_event, weight_last_event, time_difference, lambda, mu);
		} else if(strncmp(forgettingType, "exponential", 11) == 0) {
			weight_new = compute_weight_exponential(new_event, weight_last_event, time_difference, lambda, mu);
		}

		if(weight_new <= theta)
			// Is the weight lower or equal the threshold?
			// If so, it will be zeroed
			return(0);
		else {
			// This is the typical case, return the new weight
			return(weight_new);
		}
	} else {
		// Time difference was less than zero
		return(-1);
	}

}

// function writing CogSNet to file
// also weights equal zero are saved
int save_cogsnet(int numberOfNodes, int snapshotNum, float **snapshotWeightsSum, float **snapshotWeightsAvg, char pathCogSNetSum[255], char pathCogSNetAvg[255], char pathCogSNetFinalWeights[255], int *realNodeIDs, float **currentWeights) {
	FILE *filePointerSum, *filePointerAvg, * filePointerFinalWeights;
	filePointerSum = fopen(pathCogSNetSum, "w");
	filePointerAvg = fopen(pathCogSNetAvg, "w");
	filePointerFinalWeights = fopen(pathCogSNetFinalWeights, "w");

	if(filePointerSum != NULL && filePointerAvg != NULL && filePointerFinalWeights != NULL) {
		fprintf(filePointerSum, "NodeID,CogsnetVector\n");
		fprintf(filePointerAvg, "NodeID,CogsnetVector\n");
		fprintf(filePointerFinalWeights, "uid1,uid2,weight\n");

		// Write the array data to the file
		for (int i = 0; i < numberOfNodes; i++) {
			int node = realNodeIDs[i];

			fprintf(filePointerSum, "%d,\"", node);
			fprintf(filePointerAvg, "%d,\"", node);

			for (int j = 0; j < snapshotNum; j++) {
				fprintf(filePointerSum, "%f", snapshotWeightsSum[i][j]);
				fprintf(filePointerAvg, "%f", snapshotWeightsAvg[i][j]);
				if (j != snapshotNum - 1) {
					fprintf(filePointerSum, ",");
					fprintf(filePointerAvg, ",");
				}
			}
			fprintf(filePointerSum, "\"\n");
			fprintf(filePointerAvg, "\"\n");

			for (int j=0; j < numberOfNodes; j++) {
				fprintf(filePointerFinalWeights, "%d,%d,%f\n", realNodeIDs[i], realNodeIDs[j], currentWeights[i][j]);
			}
		}
	} else {
		printf("[ERROR] Saving CogSNet to %s: error opening file for writing\n\n", pathCogSNetSum);
		if (errno == ENOENT) {
            printf("Error: File not found.\n");
        } else if (errno == EACCES) {
            printf("Error: Permission denied.\n");
        } else {
            printf("Error: Failed to open the file.\n");
        }
		return 1;
	}
	
	fflush(filePointerSum);
	fflush(filePointerAvg);
	fflush(filePointerFinalWeights);

	fclose(filePointerSum);
	fclose(filePointerAvg);
	fclose(filePointerFinalWeights);

	return 0;
}

int createSnapshot(int numberOfNodes, int **snapshotInteractions, float **snapshotWeightsSum, float **snapshotWeightsAvg, int snapshotNum, int snapshotTime, char forgettingType[16], float mu, float theta, float lambda, int units, int **recentEvents, float **currentWeights, char pathCogSNetSum[255], char pathCogSNetAvg[255], int *realNodeIDs) {
	for(int i = 0; i < numberOfNodes; i++) {
		int interactionCounter = 0;

		for(int j = 0; j < numberOfNodes; j++) {
			float edgeWeight = 0;
			edgeWeight = compute_weight(snapshotTime, recentEvents[i][j], forgettingType, currentWeights[i][j], 0, mu, lambda, theta, units);

			if(snapshotInteractions[i][j] == 1) {
				snapshotWeightsSum[i][snapshotNum] += edgeWeight;
				interactionCounter++;
			}
		}

		if(interactionCounter > 0) {
			snapshotWeightsAvg[i][snapshotNum] = snapshotWeightsSum[i][snapshotNum] / interactionCounter;
		}
	}

	return 0;
}

// the main function responsible for computing CogSNet
int compute_cogsnet(int numberOfNodes, int **events, int E, int *realNodeIDs, float fSnapshotInterval, float mu, float theta, float lambda, char forgettingType[16], int units, char pathCogSNetSum[255], char pathCogSNetAvg[255], char pathCogSNetFinalWeights[255]) {
	// we declare an array for storing last events between nodes
	int **recentEvents = (int **)malloc(numberOfNodes * sizeof(int *));
	for (int i = 0; i < numberOfNodes; i++) {
        recentEvents[i] = (int *)malloc(numberOfNodes * sizeof(int));
    }

	// we declare an array for storing weights between nodes
	float **currentWeights = (float **)malloc(numberOfNodes * sizeof(float *));
	for (int i = 0; i < numberOfNodes; i++) {
        currentWeights[i] = (float *)malloc(numberOfNodes * sizeof(float));
    }

	
	// Interval between network snapshots.
	// The variable snapshotInterval is usually expressed in hours or minutes.
	// The variable units scales the snapshotInterval to seconds.
	int snapshotInterval = (int)fSnapshotInterval * units;

	// Time of the next snapshot of the network.
	// The first snapshot will be taken relative to the time of the first event in the dataset.
	int snapshotTime = events[0][2] + snapshotInterval;

	// Snapshot counter
	int snapshotsNum = 0;

	// The array for storing a complete snapshot of the network. 
	// It is saved to a file using the save_cogsnet function. 
	// Due to the potentially large size of the array, we allocate memory using malloc on the heap instead of the stack.
	float **snapshotWeightsSum = (float **)malloc(numberOfNodes * sizeof(float *));
	float **snapshotWeightsAvg = (float **)malloc(numberOfNodes * sizeof(float *));

	for (int i = 0; i < numberOfNodes; i++) {
		snapshotWeightsSum[i] = (float *)malloc(E * sizeof(float));
		snapshotWeightsAvg[i] = (float *)malloc(E * sizeof(float));
	}

	for(int i = 0; i < numberOfNodes; i++) {
		for(int j=0; j < E; j++) {
			snapshotWeightsSum[i][j] = 0;
			snapshotWeightsAvg[i][j] = 0;
		}
	}

	int **snapshotInteractions = (int **)malloc(numberOfNodes * sizeof(int *));
	for (int i = 0; i < numberOfNodes; i++) {
        snapshotInteractions[i] = (int *)malloc(numberOfNodes * sizeof(int));
    }

	// zeroing arrays
	for(int i = 0; i < numberOfNodes; i++) {
		for(int j=0; j < numberOfNodes; j++) {
			recentEvents[i][j] = 0;
			currentWeights[i][j] = 0;
			snapshotInteractions[i][j] = 0;
		}
	}
	
	// number of events that have been used for building CogSNet
	int finalNumberOfEvents = 0;

	if(snapshotInterval == 0 || ((events[E-1][2]-events[0][2])/snapshotInterval) < E) {
		// events have to be chronorogically ordered
		for(int i = 0; i < E; i++) {
			// new weight will be stored here
			double newWeight = 0;

			int uid1 = events[i][0];
			int uid2 = events[i][1];

			snapshotInteractions[uid1][uid2] = 1;
			snapshotInteractions[uid2][uid1] = 1;

			// was there any event with these uid1 and uid2 before?
			// we check it by looking at weights array, since
			// meanwhile the weight could have dropped below theta

			if(currentWeights[uid1][uid2] == 0) {
				// no events before, we set the weight to mu
				newWeight = mu;
			} else {
				// there was an event before
				newWeight = compute_weight(events[i][2], recentEvents[uid1][uid2], forgettingType, currentWeights[uid1][uid2], 1, mu, lambda, theta, units);
			}

			// set the new last event time
			// the edges are undirected, so we perform updates for both directions.
			recentEvents[uid1][uid2] = events[i][2];
			recentEvents[uid2][uid1] = events[i][2];
			
			// set the new weight
			currentWeights[uid1][uid2] = newWeight;
			currentWeights[uid2][uid1] = newWeight;

			if(snapshotInterval != 0) {
				// Take a snapshot after a specified interval has elapsed.
				// - ((i+1) < E) - check if there is a next event
				// - (snapshotTime < events[i+1][2]) - take a snapshot if the time of the next event is greater than the time of the next snapshot
				// - if the time of the snapshot and the next event is the same, the snapshot will be taken after processing that event in the next iteration of the for loop
				// - if the time between events is very distant, the while loop may take multiple snapshots.
				while(((i+1) < E) && (snapshotTime < events[i+1][2])) {
					createSnapshot(numberOfNodes, snapshotInteractions, snapshotWeightsSum, snapshotWeightsAvg, snapshotsNum, snapshotTime, forgettingType, mu, theta, lambda, units, recentEvents, currentWeights, pathCogSNetSum, pathCogSNetAvg, realNodeIDs);
					snapshotsNum++;
					snapshotTime += snapshotInterval;
					for(int i = 0; i < numberOfNodes; i++) {
						for(int j=0; j < numberOfNodes; j++) {
							snapshotInteractions[i][j] = 0;
						}
					}
				}
			} else {
				// Take a snapshot after each event
				while(((i+1) < E) && (snapshotTime < events[i+1][2])) {
					createSnapshot(numberOfNodes, snapshotInteractions, snapshotWeightsSum, snapshotWeightsAvg, snapshotsNum, snapshotTime, forgettingType, mu, theta, lambda, units, recentEvents, currentWeights, pathCogSNetSum, pathCogSNetAvg, realNodeIDs);
					snapshotsNum++;
					snapshotTime = events[i+1][2];
					for(int i = 0; i < numberOfNodes; i++) {
						for(int j=0; j < numberOfNodes; j++) {
							snapshotInteractions[i][j] = 0;
						}
					}
				}
			}

			finalNumberOfEvents++;
		}

		//  As all events are processed, we take the final snapshot of the network.
		createSnapshot(numberOfNodes, snapshotInteractions, snapshotWeightsSum, snapshotWeightsAvg, snapshotsNum, snapshotTime, forgettingType, mu, theta, lambda, units, recentEvents, currentWeights, pathCogSNetSum, pathCogSNetAvg, realNodeIDs);
		snapshotsNum++;

		int isOK = save_cogsnet(numberOfNodes, snapshotsNum, snapshotWeightsSum, snapshotWeightsAvg, pathCogSNetSum, pathCogSNetAvg, pathCogSNetFinalWeights, realNodeIDs, currentWeights);

		// Free the allocated memory
		for (int i = 0; i < numberOfNodes; i++) {
			free(recentEvents[i]);
			free(currentWeights[i]);
			free(snapshotInteractions[i]);
			free(snapshotWeightsSum[i]);
			free(snapshotWeightsAvg[i]);
		}
		free(recentEvents);
		free(currentWeights);
		free(snapshotInteractions);
		free(snapshotWeightsSum);
		free(snapshotWeightsAvg);

		return isOK;
	} else {
		printf("ERROR: Number of snapshots cannot be bigger than number of events! Increase snapshot interval.");
		return 1;
	}
}

// checks whether a given element exists in an array
int existingId(int x, int *array, int size){
	int isFound = -1;

	int i = 0;

	while(isFound < 0 && i < size){
		if(array[i] == x){
			isFound = i;
			break;
		}

		i++;
	}
	return isFound;
}

// this function returns the element in the CSV organized as three-column one (x;y;timestamp)
// it is used to extract elements both from pathSurveyDates as from pathSurveyDates

int returnElementFromCSV(char eventLine[65536], int elementNumber, char delimiter[1]) {
	char *ptr;

	ptr = strtok(eventLine, delimiter);

	int thisLineElementNumber = 0;

	while(ptr != NULL) {
	  	if(thisLineElementNumber == elementNumber) {
			return atoi(ptr);
	  	}

		thisLineElementNumber++;

		ptr = strtok(NULL, delimiter);
	}
}

int main(int argc, char** argv){
	// arguments

	// Forgetting type
	//  linear - linear forgetting
	//  power - power forgetting
	//  exponential - exponential forgetting
	char forgettingType[16];
	strcpy(forgettingType, argv[1]);

	// the snapshot interval
	float snapshotInterval = atof(argv[2]);
	
	float mu = atof(argv[3]);
	float theta = atof(argv[4]);
	float lambda = atof(argv[5]);

	// the units for forgetting, in seconds
	// usually, hours (3600)
	int units = atoi(argv[6]);

	// pathEvents with communication events
	char pathEvents[128];
	strcpy(pathEvents, argv[7]);

	// pathCogSNet with information about where to save CogSNet
	char pathCogSNetSum[255];
	strcpy(pathCogSNetSum, argv[8]);
	
	char pathCogSNetAvg[255];
	strcpy(pathCogSNetAvg, argv[9]);

	char pathCogSNetFinalWeights[255];
	strcpy(pathCogSNetFinalWeights, argv[10]);

	// the delimiter in the input/output files
	char delimiter[] = ";";

	//  the content of the pathEvents with events
	char buffer[65536];
	char *line;
	char lineCopy[65536];

	int numberOfLines = 0;
	
	// check if the file exists
	if(access(pathEvents, F_OK) != -1) {
		// define the stream for events
		FILE* filePointer;
		
		filePointer = fopen(pathEvents, "r");
			
		// check if there is no other problem with the file stream
		if(filePointer != NULL) {
			// firstly, we check how many lines we do have to read from the file

			// read the header
			line = fgets(buffer, sizeof(buffer), filePointer);

			// we read one line (header)
			numberOfLines++;

			//this will hold the event timestamp from decoded event line
			int eventTimestamp = 0;

			// now read the rest of the file until the condition won't be met
			while ((line = fgets(buffer, sizeof(buffer), filePointer)) != NULL) {

				// decode the line - get only the timestamp of the event
				eventTimestamp = returnElementFromCSV(line, 2, delimiter);

				numberOfLines++;
			}

			fclose(filePointer);

			// if the pathEvents has more then one line (header always should be there)
			if(numberOfLines > 1) {
				// define an array for events
				// sender, receiver, timestamp
				int numberOfEvents = numberOfLines - 1;
				int **events = (int **)malloc(numberOfEvents * sizeof(int *));
				for (int i = 0; i < numberOfEvents; i++) {
					events[i] = (int *)malloc(3 * sizeof(int));
				}

				// now, reopen the pathEvents with events as the stream
				filePointer = fopen(pathEvents, "r");
				
				// skip first line, as it is a header
				line = fgets(buffer, sizeof(buffer), filePointer); 
			
				// read all further lines and put them into events' matrix

				int eventNumber = 0;
				
				int eventsNodeIDSender = 0;
				int eventsNodeIDReceiver = 0;
				int eventsTimestamp = 0;

				for(eventNumber = 0; eventNumber < numberOfEvents; eventNumber++) {
					// read the line
					line = fgets(buffer, sizeof(buffer), filePointer);
					
					// extract the first element (sender's nodeID)
					strcpy(lineCopy, line);
					eventsNodeIDSender = returnElementFromCSV(lineCopy, 0, delimiter);
					
					// extract the second element (receiver's nodeID)
					strcpy(lineCopy, line);
					eventsNodeIDReceiver = returnElementFromCSV(lineCopy, 1, delimiter);

					// extract the second element (event timestamp)
					strcpy(lineCopy, line);
					eventsTimestamp = returnElementFromCSV(lineCopy, 2, delimiter);

					// set the proper values of array
					events[eventNumber][0] = eventsNodeIDSender;
					events[eventNumber][1] = eventsNodeIDReceiver;
					events[eventNumber][2] = eventsTimestamp;
				}
				
				fclose(filePointer);
				
				// the array holding real nodeIDs, sized numberOfEvents for the worst case
				int currentSize = 1;
				int *realNodeIDs = (int *)malloc(currentSize * sizeof(int));

				// actual number of nodes in the events files
				int numberOfNodes = 0;

				// ----- convert node ids -----
				for(int i = 0; i < numberOfEvents; i++){
					for(int j = 0; j < 2; j++) {
						int realNodeID = events[i][j];
						int convertedNodeID = existingId(realNodeID, realNodeIDs, numberOfNodes);

						// do we already have this nodeID in realNodeIDs?
						if(convertedNodeID < 0) {
							if(currentSize < (numberOfNodes+1)){
								currentSize *= 2; // double the size
								realNodeIDs = (int *)realloc(realNodeIDs, currentSize * sizeof(int));
							}

							// no, we need to add it
							realNodeIDs[numberOfNodes] = realNodeID;
							events[i][j] = numberOfNodes;
							numberOfNodes++;
						} else {
							events[i][j] = convertedNodeID;
						}

					}
				}

				// finally, compute CogSNet
				int isOK = compute_cogsnet(numberOfNodes, events, numberOfEvents, realNodeIDs, snapshotInterval, mu, theta, lambda, forgettingType, units, pathCogSNetSum, pathCogSNetAvg, pathCogSNetFinalWeights);
				
				// free the allocated memory
				for (int i = 0; i < numberOfEvents; i++) {
					free(events[i]);
				}
				free(events);
				free(realNodeIDs);
				
				return isOK;
			} else {
				printf("[ERROR] Reading events from %s: no events to read\n", pathEvents);
			}
		} else {
			printf("[ERROR] Reading events from %s: error reading from filestream\n", pathEvents);
		} 
	} else {
			printf("[ERROR] Reading events from %s: file not found\n", pathEvents);
	}

	return 1;
}