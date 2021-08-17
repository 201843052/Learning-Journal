CREATE TABLE Addressbook 
(regist_no INTEGER NOT NULL,
 name VARCHAR(128) NOT NULL,
 address VARCHAR(256) NOT NULL,
 tel_no CHAR(10) ,
 mail_address CHAR(20) ,
 PRIMARY KEY (regist_no));

ALTER TABLE Addressbook ADD COLUMN postal_code CHAR(8) NOT NULL;

drop table Addressbook;

-- Since dropped tables are not recoverable, creating dropped table again
CREATE TABLE Addressbook 
(regist_no INTEGER NOT NULL,
 name VARCHAR(128) NOT NULL,
 address VARCHAR(256) NOT NULL,
 tel_no CHAR(10) ,
 mail_address CHAR(20) ,
 postal_code CHAR(8) NOT NULL,
PRIMARY KEY (regist_no));
