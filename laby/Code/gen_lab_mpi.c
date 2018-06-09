#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <mpi.h>
#include <sys/time.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

// calcul du temps pris
#define DIFFTEMPS(a,b) (((b).tv_sec - (a).tv_sec) + ((b).tv_usec - (a).tv_usec)/1000000.)

/* à commenter pour désactiver l'affichage */
//#define AFFICHE

/* nombre de cases constructibles minimal */
#define CONSMIN 10
/* probabilité qu'une case du bord ne soit pas constructible */
#define PROBPASCONS 10
/* nombre d'ilots par defaut */
#define NBILOTS 2

/*probablité utilisée lors de la création des murs*/
#define PROBMUR 3

#ifdef AFFICHE
#include "graph.h"

/* taille du mur (pixels) */
#define CARRE 2
/* espace entre les pixels */
#define INTER 0
/* nombre de couleurs */
#define NBCOL 120
/* taux de rafraichissement de l'affichage */
#define REFRESH 20
/* probabilité utilisée pour la construction des lignes frontières */
#define PROBFRONTIERE 3

/* fonction qui affiche un carre de cote CARRE dans la case (i,j) */
static
void affichecarre(int i, int j)
{
	for( int k=0 ; k<CARRE ; ++k )
		line(j*(CARRE+INTER)+k,i*(CARRE+INTER),
		     j*(CARRE+INTER)+k,i*(CARRE+INTER)+CARRE-1);
}
#endif /* AFFICHE */

/* fonction estconstructible : renvoie vrai si la case (i,j) est constructible */
static
int estconstructible( size_t N, size_t M, int l[N][M], int i, int j)
{
	if( l[i][j]==0 )
		return 0;
	else if( (l[i-1][j]==0 && l[i][j+1] && l[i][j-1] && l[i+1][j-1] && l[i+1][j] && l[i+1][j+1] )
		|| (l[i+1][j]==0 && l[i][j+1] && l[i][j-1] && l[i-1][j-1] && l[i-1][j] && l[i-1][j+1] )
		|| (l[i][j-1]==0 && l[i+1][j] && l[i-1][j] && l[i-1][j+1] && l[i][j+1] && l[i+1][j+1] )
		|| (l[i][j+1]==0 && l[i+1][j] && l[i-1][j] && l[i-1][j-1] && l[i][j-1] && l[i+1][j-1] )
		)
		return 1;
	else
		return 0;

}


int main(int argc, char* argv[argc+1])
{
	struct timeval tv_beg, tv_end;
	gettimeofday( &tv_beg, NULL);
	//ADDED : MPI initialization & rank
	if (MPI_Init(&argc, &argv)) {
		perror("MPI_Init failed");
		return EXIT_FAILURE;
	}
	int rank;
	int size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	//printf("Hi! I'm proc #%d/%d! Have a nice day!\n", rank, size); // not necessary
	int i = 0, j = 0, nbilots = NBILOTS, nbcons;

#ifdef AFFICHE
	int ref=REFRESH;
#endif /* AFFICHE */

	if( argc > 1 )
		nbilots = strtoull(argv[1], 0, 0);

	/* taille du labyrinthe : */
	/* hauteur : */
	size_t N = 400;
	size_t N_save = 400;
	if( argc > 2 )
	{
		N = strtoull(argv[2], 0, 0);
		N_save = strtoull(argv[2], 0, 0);
	}
	/* largeur : */
	size_t M = 600;
	size_t M_save = 600;
	if( argc > 3 )
	{
		M = strtoull(argv[3], 0, 0); // ADDED correction : ligne originale: "N=strtoull..."
		M_save = strtoull(argv[3], 0, 0);
	}
	int (*l)[M] = malloc(sizeof(int[N][M]));

	/* initialise l : murs autour, vide a l'interieur */
	// ADDED: seul un proc a besoin de créer les mur autour du tableau initial
	if(rank == 0 ){
		for( i=0 ; i<N ; i++ )
			for( j=0 ; j<M ; j++ )
			if( i==0 || i==N-1 ||j==0 || j==M-1 )
			{
				l[i][j] = 0; /* mur */
			}
			else
				l[i][j] = 1; /* vide */

		// Initialisation des frontières
		for( i=1 ; i<size ; i++ )
			for( j=0 ; j<M ; j++ )
				if (rand()%PROBMUR != 0 )
					l[N/size*i][j] = 0;
	}

	/* ADDED buffer pour stocker la partie du tableau dont le processus est reponsable */
	/* Repartition du tableau entre les différents procs */
	int(*l2)[M] = NULL;

	/* Si le processus est le dernier processus, il recupère tout ce qu'il reste */
	if(rank != (size-1)) {
		l2 = malloc(sizeof(int[N/size+1][M]));
		N = N/size+1;
	}
	else {
		l2 = malloc(sizeof(int[N/size+(N%size)][M]));
		N=N/size+(N%size);
	}

	// ADDED: initilisation des tableaus utilisés pour scatterv et gatherv
	// la variable N a déjà été modifiée par le processus pour représenter sa p
	// sa propre dimension. La hauteur du tableau a été sauvegardée
	// préalablement dans N_save
	// Chaque processus récupère une ligne supplémentaire.
	// ie la dernière ligne du processus X est la première ligne du processus X+1
	// Cette duplication est corrigée lors du gather (ligne écrasée)
	// Ce comportement peut être implémenté car displ[] permet de renseigner
	// l'offset des données à récupérer selon l'origine du tableau
	
	int counts[size]; //nombre d'éléments pour chaque chunk
	int displ[size]; //offset auquel récupérer les élements du chunk
	for( i=0; i <size; i++ ){
		counts[i] = N*M;
		displ[i]  = i*(N_save/size)*M;
		if(i==size-1)
		{
			counts[i] = ((N_save/size+(N_save%size)))*M;
			displ[i] = i*(N_save/size)*M;
		}
	}

	//ADDED : répartition des données
	MPI_Scatterv(
		l,
		counts,
		displ,
		MPI_INTEGER,
		l2,
		N*M,
		MPI_INTEGER,
		0,
		MPI_COMM_WORLD);



#ifdef AFFICHE
	initgraph(M_save*(CARRE+INTER), N_save*(CARRE+INTER));
	for( int i=0 ; i<N ; i++ )
		for( int j=0 ; j<M ; j++ )
			if( l2[i][j]==0 )
				affichecarre(i,j);
	refresh();
#endif /* AFFICHE */

	/* place <nbilots> ilots aleatoirement a l'interieur de chaque section */
	/* ADDED: changement de seed pour que chaque proc tire des nombres différents */
	/* ( ajout du rank du proc comme variable de la seed ) */
	srand( time(0) + rank*rank*size );
	for( nbilots=NBILOTS; nbilots ; nbilots-- )
	{
		i = rand()%(N-4) + 2;
		j = rand()%(M-4) + 2;
		l2[i][j] = 0;
	}


	/* Chaque proc travaille sur son tableau l2 */
	/* initialisation des cases constructibles */
	nbcons = 0;
	for( int i=1 ; i<N-1 ; i++ )
		for( int j=1 ; j<M-1 ; j++ )
			if( estconstructible( N, M, l2, i, j ) )
			{
				l2[i][j] = -1;
				nbcons++;
			}

	// supprime quelques cases constructibles sur les bords
	/* chaque procs doit supprimer des cases constructibles sur les bords
		 gauche et droit */
	/* Seul le premier et dernier proc doivent supprimer des cases constructibles
	   sur les extrémités supérieure et inférieure (car division du tableau par rang
	 		selon la doc MPI) */

	// Supression sur les bords gauche et droit
	for( int i=1 ; i<N-1 ; i++ )
	{
		if( l2[i][1] == -1 && (rand()%PROBPASCONS) && nbcons>(CONSMIN*2) )
		{
			l2[i][1] = 1;
			nbcons--;
		}
		if( l2[i][M-2] == -1 && (rand()%PROBPASCONS) && nbcons>(CONSMIN*2) )
		{
			l2[i][M-2] = 1;
			nbcons--;
		}
	}

	// Suppression bord supérieur et inférieur
	if( rank == 0 )
	{
		for( int j=1 ; j<M-1 ; j++ )
		{
			if( l2[1][j] == -1 && (rand()%PROBPASCONS) && nbcons>CONSMIN )
			{
				l2[1][j] = 1;
				nbcons--;
			}
		}
	}

	// bord inférieur
	if( rank == size )
	{
		for( int j=1 ; j<M-1 ; j++ )
		{
			if( l2[N-2][j] == -1 && (rand()%PROBPASCONS) && nbcons>CONSMIN )
			{
				l2[N-2][j] = 1;
				nbcons--;
			}
		}
	}

	// boucle principale de génération
	while( nbcons )
	{
		int r = 1 + rand() % nbcons;
		for( i=1 ; i<N-1 ; i++ )
		{
			for( j=1 ; j<M-1 ; j++ )
				if( l2[i][j] == -1 )
					if( ! --r )
						break;
			if( ! r )
				break;
		}
		//on construit en (i,j)
		l2[i][j] = 0;

#ifdef AFFICHE
		affichecarre(i,j);
		if( ! --ref )
		{
			refresh();
			ref = REFRESH;
		}
#endif // AFFICHE

		nbcons --;
		// met a jour les 8 voisins
		for( int ii=i-1 ; ii<=i+1 ; ++ii )
			for( int jj=j-1 ; jj<=j+1 ; ++jj )
				if( l2[ii][jj]==1 && estconstructible(N, M, l2, ii,jj) )
				{
					nbcons ++;
					l2[ii][jj] = -1;
				}
				else if( l2[ii][jj]==-1 && ! estconstructible(N, M, l2, ii,jj) )
				{
					nbcons --;
					l2[ii][jj] = 1;
				}
	}	// fin while */

	//ADDED : récupération des sections
	MPI_Gatherv(
		l2,
		N*M,
		MPI_INTEGER,
		l,
		counts,
		displ,
		MPI_INTEGER,
		0,
		MPI_COMM_WORLD);
	gettimeofday( &tv_end, NULL);

  /* Seul un proc affiche le résultat final et  enregistre le fichier */
	if(rank == 0)
	{
		printf( "%lux%lu\t%lf\n",N_save,M_save, DIFFTEMPS(tv_beg,tv_end));		
		#ifdef AFFICHE
		initgraph(M_save*(CARRE+INTER), N_save*(CARRE+INTER));
		for( int i=0 ; i<N_save ; i++ )
			for( int j=0 ; j<M_save ; j++ )
			{				
				if( l[i][j] == 0 )
					affichecarre(i,j);
			}
		refresh();
		#endif /* AFFICHE */

	/* ENREGISTRE UN FICHIER. Format : LARGEUR(int), HAUTEUR(int), tableau brut (N*M (int))*/
		int f = open( "laby.lab", O_WRONLY|O_CREAT, 0644 );
		int x = N_save;
		int no_warning = 0;
		no_warning = write( f, &x, sizeof(int) );
		x = M_save;
		no_warning = write( f, &x, sizeof(int) );
		no_warning = write( f, l, N_save*M_save*sizeof(int) );
		no_warning++;
		close( f );

		#ifdef AFFICHE
		refresh();
		waitgraph(); /* attend que l'utilisateur tape une touche */
		closegraph();
		#endif /* AFFICHE */

 	}


	// Keepin' it clean
	free(l);
	free(l2);
	MPI_Finalize();
	return EXIT_SUCCESS;
}
