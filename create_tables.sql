drop table chunks;

create table CHUNKS 
(ID VARCHAR2(64) NOT NULL,
CHUNK CLOB,
PRIMARY KEY ("ID")
);

drop table vectors;

create table VECTORS
("ID" VARCHAR2(64) NOT NULL,
"VEC" VECTOR(1024, FLOAT64),
PRIMARY KEY ("ID")
);

