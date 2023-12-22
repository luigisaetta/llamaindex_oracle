drop table BOOKS;
  
create table BOOKS
("ID" NUMBER NOT NULL,
"NAME" VARCHAR2(100) NOT NULL,
PRIMARY KEY ("ID")  
);

drop table chunks;

create table CHUNKS 
(ID VARCHAR2(64) NOT NULL,
CHUNK CLOB,
BOOK_ID NUMBER,
PRIMARY KEY ("ID"),
CONSTRAINT fk_book
        FOREIGN KEY (BOOK_ID)
        REFERENCES BOOKS (ID)
);

drop table vectors;

create table VECTORS
("ID" VARCHAR2(64) NOT NULL,
"VEC" VECTOR(1024, FLOAT64),
PRIMARY KEY ("ID")
);



